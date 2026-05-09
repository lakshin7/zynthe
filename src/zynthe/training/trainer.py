import copy
import csv
from datetime import datetime
from zynthe.evaluation.metrics import compute_all_metrics, plot_metrics
from zynthe.evaluation.visualizer import (
    plot_training_curves,
    plot_teacher_student_comparison,
    plot_epoch_micro_series,
    plot_evaluation_dashboard,
    plot_distillation_gap,
    plot_extended_metrics,
)
from zynthe.evaluation.metrics_extended import (
    compute_extended_metrics,
    LossComponentTracker,
)
from zynthe.evaluation.evaluation_report import EvaluationReport
from zynthe.evaluation.diagnostics import build_eval_diagnostics
from zynthe.core.models.model_saver import ModelSaver
from zynthe.training.optimizer import OptimizerFactory, GradientManager, AdaptiveOptimizer
from zynthe.training.scheduler import SchedulerFactory
from zynthe.core.utils.data_validator import DataValidator, OverfitUnderfitDetector
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.amp import GradScaler  # Mixed Precision Training
import os
import inspect
import json
import time
import logging
from pathlib import Path
from typing import Optional, Callable, Dict, Any, List
import numpy as np

LOG = logging.getLogger(__name__)

class Trainer:
    def __init__(
        self,
        teacher,
        student,
        tokenizer,
        config,
        device,
        experiment_dir,
        websocket_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        pipeline=None,  # NEW: Optional pipeline parameter
    ):
        self.teacher = teacher
        self.student = student
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        self.experiment_dir = experiment_dir
        self._use_pipeline = pipeline is not None  # Track if using pipeline mode
        
        # ========== TEACHER MODEL VALIDATION ==========
        # Detect if teacher is a base (untrained) model vs task-specific fine-tuned model
        # Base models (e.g., bert-base-uncased) won't perform well on downstream tasks
        # and will produce poor distillation quality
        teacher_model_name = getattr(teacher.config, '_name_or_path', 'unknown')
        is_base_model = any(indicator in teacher_model_name.lower() for indicator in [
            'base-uncased', 'base-cased', 'base-multilingual',
            '-base', 'gpt2', 't5-small', 't5-base'
        ])
        
        if is_base_model:
            LOG.warning("=" * 80)
            LOG.warning("  TEACHER MODEL WARNING: Base model detected!")
            LOG.warning(f"   Teacher: {teacher_model_name}")
            LOG.warning("   Base models are NOT task-trained and will perform poorly!")
            LOG.warning("   This leads to low-quality distillation (teacher < student).")
            LOG.warning("")
            LOG.warning("   RECOMMENDATIONS:")
            LOG.warning("   1. Use a task-specific fine-tuned teacher model, OR")
            LOG.warning("   2. Enable teacher training: set train_teacher: true in config")
            LOG.warning("=" * 80)
            print(f"\n  WARNING: Using base model '{teacher_model_name}' as teacher.")
            print("   This may result in poor distillation. Consider using a fine-tuned model.")
            print("   See logs for recommendations.\n")
        
        # ========== END TEACHER MODEL VALIDATION ==========
        
        # ========== PERFORMANCE OPTIMIZATIONS ==========
        train_cfg = self.config.get('train', {})
        
        # 1. Mixed Precision Training (AMP) - 2-3x speedup
        self.use_amp = bool(train_cfg.get('use_amp', train_cfg.get('mixed_precision', True)))
        scaler_device = 'cuda' if device.type == 'cuda' else 'cpu'
        self.scaler = GradScaler(device=scaler_device, enabled=self.use_amp) if self.use_amp else None
        if self.use_amp:
            print("[OPTIMIZATION] Mixed Precision Training (AMP) enabled - expect 2-3x speedup")
        
        # 2. Gradient Accumulation - simulate larger batch sizes
        self.gradient_accumulation_steps = int(train_cfg.get('gradient_accumulation_steps', train_cfg.get('grad_accum_steps', 1)))
        if self.gradient_accumulation_steps > 1:
            print(f"[OPTIMIZATION] Gradient Accumulation enabled ({self.gradient_accumulation_steps} steps) - effective batch size x{self.gradient_accumulation_steps}")
        
        # 3. Model Compilation - DISABLED for type safety
        # torch.compile() is powerful but causes type inference issues
        # Users can manually compile models before passing to Trainer if needed
        # Example: model = torch.compile(model, mode='reduce-overhead')
        
        # 4. Live Metrics Streaming via WebSocket - real-time UI updates
        self.websocket_callback = websocket_callback
        self.update_frequency = self.config['train'].get('update_frequency', 10)  # Update every N batches
        if self.websocket_callback:
            print(f"[OPTIMIZATION] Live metrics streaming enabled (update every {self.update_frequency} batches)")
        
        # 5. Mac M2 specific optimizations
        if device.type == 'mps':
            print("[OPTIMIZATION] Mac M2 MPS backend detected - applying optimizations")
            # Set MPS memory fraction to avoid OOM
            if hasattr(torch.mps, 'set_per_process_memory_fraction'):
                torch.mps.set_per_process_memory_fraction(0.8)
            # Enable fallback to CPU for unsupported ops
            os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
        
        # ========== END PERFORMANCE OPTIMIZATIONS ==========
        
        # Create optimizer using OptimizerFactory (phase-aware, gradient management)
        self.optimizer = OptimizerFactory.get_optimizer(
            self.student,
            self.config['train'],
            phase='distillation'
        )
        
        # Create learning rate scheduler
        # Estimate steps per epoch (will be updated in train())
        self.scheduler = None  # Will be initialized in train() with actual steps_per_epoch
        
        # Create adaptive optimizer wrapper (DEI/CAS-based LR tuning)
        enable_adaptive = self.config['train'].get('dynamic_lr', True)
        self.adaptive_opt = AdaptiveOptimizer(
            self.optimizer,
            enable_auto_tune=enable_adaptive
        )
        
        # Teacher training configuration
        self.should_train_teacher = self.config['train'].get('train_teacher', False)
        self.teacher_epochs = self.config['train'].get('teacher_epochs', 2)
        if self.should_train_teacher:
            raw_teacher_lr = self.config['train'].get('teacher_lr', 2e-5)
            try:
                teacher_lr = float(raw_teacher_lr)
            except (TypeError, ValueError):
                raise ValueError(f"Invalid train.teacher_lr value: {raw_teacher_lr!r}")
            self.teacher_optimizer = AdamW(self.teacher.parameters(), lr=teacher_lr)
        
        # ========== PIPELINE / DISTILLER INITIALIZATION ==========
        # Pipeline-first design: ALL distillation flows go through a pipeline.
        # If a legacy distiller is configured, it gets wrapped in a
        # SingleDistillerPipeline automatically.  This eliminates the
        # previous dual-path branching throughout the training loop.
        
        if pipeline is not None:
            # Pipeline passed directly by caller
            self.pipeline = pipeline
            print(f"[TRAINER] Using pipeline: {pipeline.name}")
        else:
            distil_cfg = self.config.get('distillation', {}) or {}
            pipeline_cfg = distil_cfg.get('pipeline', {})
            
            if pipeline_cfg and pipeline_cfg.get('type') in ['multi_stage', 'multistage', 'multi']:
                # Config requests a multi-stage pipeline
                print("[TRAINER] Building pipeline from configuration...")
                from zynthe.core.pipelines import PipelineBuilder
                self.pipeline = PipelineBuilder.from_config(
                    self.config,
                    self.teacher,
                    self.student,
                    self.device
                )
                print(f"[TRAINER] Pipeline built: {self.pipeline.name}")
            else:
                # Legacy distiller → auto-wrap in SingleDistillerPipeline
                from zynthe.core.distillers.multi_stage_distiller import DistillerRegistry  # noqa: F811
                from zynthe.core.pipelines.single_distiller_pipeline import SingleDistillerPipeline
                
                registry = DistillerRegistry()
                distiller_type = distil_cfg.get('method') or distil_cfg.get('type') or 'kd_hinton'
                distiller_aliases = {
                    'hinton': 'kd_hinton',
                    'kdhinton': 'kd_hinton',
                    'feature_distillation': 'feature',
                    'similarity_transfer': 'similarity',
                    'attention_transfer': 'attention',
                }
                distiller_type = distiller_aliases.get(
                    str(distiller_type).lower(), str(distiller_type).lower()
                )
                self.distiller_type = distiller_type
                
                distiller_class = registry.get(distiller_type)
                if distiller_class is None:
                    raise ValueError(f"Unknown distiller type: {distiller_type}")
                
                distiller_config = copy.deepcopy(distil_cfg.get('config', {}))
                for key in ('temperature', 'alpha', 'label_smoothing',
                            'hint_enabled', 'confidence_scaling', 'min_confidence'):
                    if key in distil_cfg and key not in distiller_config:
                        distiller_config[key] = distil_cfg[key]
                if isinstance(distil_cfg.get('kd_hinton'), dict) and 'kd_hinton' not in distiller_config:
                    distiller_config['kd_hinton'] = copy.deepcopy(distil_cfg['kd_hinton'])
                
                # Instantiate distiller using signature introspection
                _distiller_sig = inspect.signature(distiller_class.__init__)  # type: ignore[misc]
                _distiller_params = set(_distiller_sig.parameters.keys()) - {'self'}
                
                if 'config' in _distiller_params:
                    distiller = distiller_class(
                        teacher=self.teacher,
                        student=self.student,
                        config=distiller_config,
                        device=self.device,
                    )
                else:
                    distiller = distiller_class(
                        teacher=self.teacher,
                        student=self.student,
                        device=self.device,
                        **distiller_config,
                    )
                
                # Wrap in pipeline for unified interface
                self.pipeline = SingleDistillerPipeline(
                    distiller,
                    name=f"AutoWrapped_{distiller_class.__name__}",
                )
                print(f"[TRAINER] Distiller '{distiller_type}' auto-wrapped in pipeline")
        
        # Ensure pipeline is set up
        if not self.pipeline._is_setup:
            self.pipeline.setup()
            self.pipeline._is_setup = True
        
        # Pipeline-first: always True now.
        self._use_pipeline = True
        
        # Determine if attention outputs are needed
        self._requires_attention_outputs = self._distiller_needs_attention_outputs()
        # ========== END PIPELINE / DISTILLER INITIALIZATION ==========
        
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.metrics_history: Dict[str, List[float]] = {'accuracy': [], 'f1': [], 'precision': [], 'recall': []}
        # Optional batch-level tracking for detailed visualization
        self.batch_train_losses: List[List[float]] = []  # List[List[float]] per epoch
        self.batch_val_losses: List[List[float]] = []    # List[List[float]] per epoch
        self.batch_val_running_acc: List[List[float]] = []  # List[List[float]] per epoch
        self.best_val_loss = float('inf')
        self.resume_epoch = 0
        self.resume_global_step = 0
        self.best_model_state = None
        self.early_stop_patience = self.config['train'].get('early_stop_patience', 2)
        self.no_improve_epochs = 0
        self.last_preds: List[Any] = []
        self.last_labels: List[Any] = []

        # Overfitting guard configuration ensures we automatically react if
        # validation loss starts diverging sharply from training loss.
        raw_guard_cfg = self.config['train'].get('overfit_guard', True)
        if isinstance(raw_guard_cfg, dict):
            guard_enabled = raw_guard_cfg.get('enabled', True)
            guard_min_epochs = int(raw_guard_cfg.get('min_epochs', 3) or 3)
            guard_mode = raw_guard_cfg.get('mode', raw_guard_cfg.get('action', 'early_stop'))
            guard_conf_threshold = float(raw_guard_cfg.get('confidence_threshold', 0.5))
        else:
            guard_enabled = bool(raw_guard_cfg)
            guard_min_epochs = 3
            guard_mode = 'early_stop'
            guard_conf_threshold = 0.5

        self.overfit_guard_config = {
            'enabled': guard_enabled,
            'min_epochs': max(2, guard_min_epochs),
            'mode': guard_mode,
            'confidence_threshold': max(0.0, min(1.0, guard_conf_threshold))
        }
        self.overfit_guard_state: Dict[str, Any] = {
            'triggered': False,
            'epoch': None,
            'status': None,
            'confidence': 0.0,
            'loss_gap_pct': 0.0,
            'analysis': None,
            'action': None,
            'events': []
        }

        raw_mitigation_cfg = self.config['train'].get('overfit_mitigation', True)
        mitigation_defaults = {
            'enabled': True,
            'max_interventions': 4,
            'cooldown_epochs': 1,
            'lr_factor': 0.7,
            'lr_min': 1e-6,
            'weight_decay_floor': 0.02,
            'enable_augmentation': True,
            'enable_dropout': True,
            'enable_weight_decay': True,
            'enable_lr_decay': True,
            'augment_apply_prob': 0.45,
            'augment_dropout': 0.15,
            'augment_noise': 0.05,
            'dropout_increment': 0.1,
            'dropout_max': 0.5
        }

        if isinstance(raw_mitigation_cfg, dict):
            mitigation_enabled = raw_mitigation_cfg.get('enabled', True)
            mitigation_config = mitigation_defaults.copy()
            for key in mitigation_defaults.keys():
                if key in raw_mitigation_cfg and key != 'enabled':
                    mitigation_config[key] = raw_mitigation_cfg[key]
            mitigation_config['enabled'] = mitigation_enabled
        elif raw_mitigation_cfg in (False, None):
            mitigation_config = mitigation_defaults.copy()
            mitigation_config['enabled'] = False
        else:
            mitigation_config = mitigation_defaults.copy()

        self.overfit_mitigation_config = mitigation_config
        self.overfit_mitigation_state: Dict[str, Any] = {
            'intervention_count': 0,
            'history': [],
            'last_epoch': None,
            'counts': {
                'augmentation': 0,
                'dropout': 0,
                'weight_decay': 0,
                'lr': 0
            }
        }
        self._baseline_optimizer_lrs: List[float] = []
        self._baseline_weight_decays: List[float] = []
        self._train_loader_ref = None

        # Teacher tracking buffers are always initialised so downstream summaries
        # can safely inspect them even when teacher fine-tuning is disabled.
        self.teacher_epoch_losses: List[float] = []
        self.teacher_epoch_val_losses: List[float] = []
        self.teacher_batch_train_losses: List[List[float]] = []
        self.teacher_batch_val_losses: List[List[float]] = []
        self.teacher_batch_val_running_acc: List[List[float]] = []
        self.teacher_last_preds: List[Any] = []
        self.teacher_last_labels: List[Any] = []
        
        # Extended metrics tracking
        self.loss_tracker = LossComponentTracker()
        self.extended_metrics_history: Dict[str, List[float]] = {
            'kl_divergence': [],
            'js_divergence': [],
            'prediction_agreement': [],
            'confidence_correlation': []
        }
        self.metrics_detail_history: List[Dict[str, Any]] = []
        self.eval_runtime_history: List[Dict[str, Any]] = []
        self.eval_diagnostics_history: List[Dict[str, Any]] = []
        self.eval_calibration_history: List[Dict[str, Any]] = []
        
        # Cache model forward signatures for efficient parameter filtering
        self._teacher_forward_params = self._get_forward_params(self.teacher)
        self._student_forward_params = self._get_forward_params(self.student)

        # ========== DETAILED LOGGING OPTIONS ==========
        train_cfg = self.config.get('train', {})
        self.log_detail: bool = train_cfg.get('log_detail', True)  # master switch
        self.batch_log_interval: int = int(train_cfg.get('batch_log_interval', 10))
        self.csv_logging: bool = train_cfg.get('csv_logging', True)
        self.enable_comparison_plot: bool = train_cfg.get('enable_comparison_plot', True)
        self.show_eta: bool = train_cfg.get('show_eta', True)
        self._csv_path = os.path.join(self.experiment_dir, 'training_detailed_log.csv')
        self._csv_fieldnames = [
            'timestamp','phase','epoch','batch_idx','batches_total','loss','scaled_loss','lr','grad_norm',
            'running_acc','throughput_samples_per_s','elapsed_s','eta_s','is_teacher'
        ]
        self._csv_writer = None
        if self.csv_logging:
            try:
                init_file = not os.path.exists(self._csv_path)
                self._csv_file_handle = open(self._csv_path, 'a', newline='')
                self._csv_writer = csv.DictWriter(self._csv_file_handle, fieldnames=self._csv_fieldnames)
                if init_file:
                    self._csv_writer.writeheader()
                print(f"[LOG] Detailed batch CSV logging enabled -> {self._csv_path}")
            except Exception as e:
                print(f"[WARNING] Failed to initialize CSV logger: {e}")
                self.csv_logging = False

        # Teacher metrics history for comparison plot later
        self.teacher_metrics_history: Dict[str, List[float]] = {'accuracy': [], 'f1': [], 'precision': [], 'recall': []}

    # ------------------------------------------------------------------
    # Internal helper: Batch logging (console + optional CSV)
    # ------------------------------------------------------------------
    def _log_batch(self,
                   phase: str,
                   epoch: int,
                   batch_idx: int,
                   batches_total: int,
                   loss: float,
                   scaled_loss: float,
                   lr: float,
                   grad_norm: Optional[float],
                   running_acc: Optional[float],
                   start_time_epoch: float,
                   samples_processed: int,
                   batch_size: int,
                   is_teacher: bool):
        if not self.log_detail:
            return
        # Compute elapsed and ETA
        elapsed = time.time() - start_time_epoch
        avg_per_batch = elapsed / max(batch_idx + 1, 1)
        remaining_batches = batches_total - (batch_idx + 1)
        eta_s = avg_per_batch * remaining_batches if self.show_eta else 0.0
        throughput = samples_processed / max(elapsed, 1e-6)
        grad_norm_str = f"{grad_norm:.3f}" if (grad_norm is not None and grad_norm > 0) else 'N/A'
        acc_str = f"{running_acc*100:.2f}%" if running_acc is not None else 'N/A'
        console_msg = (
            f"[{phase.upper():7}] Ep {epoch} Batch {batch_idx+1:04d}/{batches_total} "
            f"Loss={loss:.4f} (scaled={scaled_loss:.4f}) LR={lr:.2e} Grad={grad_norm_str} Acc={acc_str} "
            f"Throughput={throughput:.1f} samp/s Elapsed={elapsed:.1f}s"
        )
        if self.show_eta:
            console_msg += f" ETA={eta_s:.1f}s"
        console_msg += f" {'[TEACHER]' if is_teacher else '[STUDENT]'}"
        if (batch_idx + 1) % self.batch_log_interval == 0 or batch_idx == 0 or (batch_idx + 1) == batches_total:
            print(console_msg)
        # CSV row
        if self._csv_writer:
            try:
                self._csv_writer.writerow({
                    'timestamp': datetime.utcnow().isoformat(),
                    'phase': phase,
                    'epoch': epoch,
                    'batch_idx': batch_idx + 1,
                    'batches_total': batches_total,
                    'loss': f"{loss:.6f}",
                    'scaled_loss': f"{scaled_loss:.6f}",
                    'lr': f"{lr:.6e}",
                    'grad_norm': grad_norm if grad_norm is not None else '',
                    'running_acc': running_acc if running_acc is not None else '',
                    'throughput_samples_per_s': f"{throughput:.3f}",
                    'elapsed_s': f"{elapsed:.3f}",
                    'eta_s': f"{eta_s:.3f}" if self.show_eta else '',
                    'is_teacher': int(is_teacher)
                })
            except Exception as e:
                LOG.warning(f"Failed to write CSV batch log: {e}")

    def _init_mitigation_context(self, train_loader) -> None:
        self._train_loader_ref = train_loader
        self._baseline_optimizer_lrs = [group.get('lr', 0.0) for group in self.optimizer.param_groups]
        self._baseline_weight_decays = [group.get('weight_decay', 0.0) for group in self.optimizer.param_groups]
        dataset = getattr(train_loader, 'dataset', None)
        if dataset is not None and hasattr(dataset, 'augmenter'):
            self.overfit_mitigation_state['baseline_augmentation'] = bool(getattr(dataset, 'augmenter', None))
        else:
            self.overfit_mitigation_state['baseline_augmentation'] = False

    def _apply_overfit_mitigation(self, epoch_idx: int, analysis: Dict[str, Any], train_loader) -> List[str]:
        cfg = self.overfit_mitigation_config
        if not cfg.get('enabled', True):
            return []
        state = self.overfit_mitigation_state
        max_interventions = int(cfg.get('max_interventions', 0))
        if max_interventions and state.get('intervention_count', 0) >= max_interventions:
            return []
        cooldown = int(cfg.get('cooldown_epochs', 0))
        last_epoch = state.get('last_epoch')
        if last_epoch is not None and cooldown > 0 and (epoch_idx - last_epoch) <= cooldown:
            return []

        actions = []
        if cfg.get('enable_augmentation', True):
            actions.append(lambda: self._mitigation_enable_augmentation(train_loader))
        if cfg.get('enable_dropout', True):
            actions.append(self._mitigation_increase_dropout)
        if cfg.get('enable_weight_decay', True):
            actions.append(self._mitigation_raise_weight_decay)
        if cfg.get('enable_lr_decay', True):
            actions.append(self._mitigation_reduce_lr)

        for action_fn in actions:
            result = action_fn()
            if result:
                state['intervention_count'] = state.get('intervention_count', 0) + 1
                state['last_epoch'] = epoch_idx
                state.setdefault('history', []).append({
                    'epoch': epoch_idx,
                    'action': result,
                    'status': analysis.get('status'),
                    'confidence': analysis.get('confidence'),
                    'loss_gap_pct': analysis.get('loss_gap_pct'),
                })
                if isinstance(result, str):
                    return [result]
                return list(result)
        return []

    def _mitigation_enable_augmentation(self, train_loader) -> Optional[str]:
        if train_loader is None:
            return None
        dataset = getattr(train_loader, 'dataset', None)
        if dataset is None or not hasattr(dataset, 'augmenter'):
            return None
        try:
            from zynthe.data.augmentations import AugmentationConfig, TextAugmenter  # type: ignore
        except Exception:
            return None

        counts = self.overfit_mitigation_state.setdefault('counts', {})
        applied = counts.get('augmentation', 0)
        max_boosts = 3
        if applied >= max_boosts:
            return None

        cfg = self.overfit_mitigation_config
        base_prob = float(cfg.get('augment_apply_prob', 0.45))
        base_dropout = float(cfg.get('augment_dropout', 0.15))
        base_noise = float(cfg.get('augment_noise', 0.05))
        augmenter = getattr(dataset, 'augmenter', None)

        if augmenter is None or not getattr(augmenter, 'config', None) or not augmenter.config.enable:
            aug_config = AugmentationConfig(
                enable=True,
                apply_prob=min(0.95, base_prob),
                dropout_prob=min(0.9, base_dropout),
                swap_prob=0.05,
                synonym_prob=0.1,
                noise_prob=min(0.5, base_noise),
                max_ops_per_sample=3,
                reserved_tokens=(),
                min_tokens=3,
                apply_to_fields=("text",),
                random_seed=None,
            )
            dataset.augmenter = TextAugmenter(aug_config)
            counts['augmentation'] = applied + 1
            return f"augmentation_enabled(p={aug_config.apply_prob:.2f})"

        config = augmenter.config
        boost_prob = min(0.95, config.apply_prob + 0.1)
        boost_dropout = min(0.9, max(config.dropout_prob, base_dropout) + 0.05)
        boost_noise = min(0.5, max(config.noise_prob, base_noise) + 0.02)

        if (
            abs(boost_prob - config.apply_prob) < 1e-6
            and abs(boost_dropout - config.dropout_prob) < 1e-6
            and abs(boost_noise - config.noise_prob) < 1e-6
        ):
            return None

        aug_config = AugmentationConfig(
            enable=True,
            apply_prob=boost_prob,
            dropout_prob=boost_dropout,
            swap_prob=config.swap_prob,
            synonym_prob=config.synonym_prob,
            noise_prob=boost_noise,
            max_ops_per_sample=config.max_ops_per_sample,
            reserved_tokens=config.reserved_tokens,
            min_tokens=config.min_tokens,
            apply_to_fields=config.apply_to_fields,
            random_seed=config.random_seed,
        )
        dataset.augmenter = TextAugmenter(aug_config)
        counts['augmentation'] = applied + 1
        return f"augmentation_boost(p={boost_prob:.2f})"

    def _mitigation_increase_dropout(self) -> Optional[str]:
        student_config = getattr(self.student, 'config', None)
        if student_config is None:
            return None
        cfg = self.overfit_mitigation_config
        inc = float(cfg.get('dropout_increment', 0.1))
        max_p = float(cfg.get('dropout_max', 0.5))
        counts = self.overfit_mitigation_state.setdefault('counts', {})

        changed = False
        for attr in (
            'hidden_dropout_prob',
            'attention_probs_dropout_prob',
            'classifier_dropout',
            'embedding_dropout_prob',
        ):
            if hasattr(student_config, attr):
                current = float(getattr(student_config, attr))
                target = min(max_p, current + inc)
                if target > current + 1e-6:
                    setattr(student_config, attr, target)
                    changed = True

        for module in self.student.modules():
            if isinstance(module, nn.Dropout):
                current_p = float(module.p)
                target_p = min(max_p, current_p + inc)
                if target_p > current_p + 1e-6:
                    module.p = target_p
                    changed = True

        if not changed:
            return None

        counts['dropout'] = counts.get('dropout', 0) + 1
        return f"dropout_step(+{inc:.2f})"

    def _mitigation_raise_weight_decay(self) -> Optional[str]:
        if not self.optimizer or not self.optimizer.param_groups:
            return None
        cfg = self.overfit_mitigation_config
        floor = float(cfg.get('weight_decay_floor', 0.02))
        updated = False
        for group in self.optimizer.param_groups:
            if 'weight_decay' not in group:
                continue
            current = float(group.get('weight_decay', 0.0))
            if current == 0.0:
                continue
            if current < floor:
                group['weight_decay'] = floor
                updated = True

        if not updated:
            return None

        counts = self.overfit_mitigation_state.setdefault('counts', {})
        counts['weight_decay'] = counts.get('weight_decay', 0) + 1
        return f"weight_decay_floor({floor:.4f})"

    def _mitigation_reduce_lr(self) -> Optional[str]:
        if not self.optimizer or not self.optimizer.param_groups:
            return None
        cfg = self.overfit_mitigation_config
        factor = float(cfg.get('lr_factor', 0.7))
        if factor >= 0.999:
            return None
        min_lr = float(cfg.get('lr_min', 1e-6))
        updated = False
        for group in self.optimizer.param_groups:
            if 'lr' not in group:
                continue
            current = float(group.get('lr', 0.0))
            if current <= min_lr + 1e-12:
                continue
            new_lr = max(min_lr, current * factor)
            if new_lr < current - 1e-12:
                group['lr'] = new_lr
                updated = True

        if not updated:
            return None

        if self.scheduler is not None and hasattr(self.scheduler, 'base_lrs'):
            try:
                setattr(self.scheduler, 'base_lrs', [group.get('lr', 0.0) for group in self.optimizer.param_groups])
            except Exception:
                pass

        counts = self.overfit_mitigation_state.setdefault('counts', {})
        counts['lr'] = counts.get('lr', 0) + 1
        return f"lr_decay(factor={factor:.2f})"

    def _get_forward_params(self, model):
        """Get the parameter names accepted by the model's forward method."""
        try:
            forward_signature = inspect.signature(model.forward)
            return set(forward_signature.parameters.keys())
        except Exception:
            # Fallback: assume standard transformer parameters
            return {'input_ids', 'attention_mask', 'labels'}
    
    def _filter_batch_for_model(self, batch, model_params):
        """Filter batch to only include parameters accepted by the model."""
        filtered_batch = {}
        for key, value in batch.items():
            if key in model_params:
                filtered_batch[key] = value
            else:
                # Log first few times to help debug, then suppress
                if not hasattr(self, '_param_warnings'):
                    self._param_warnings = set()
                if key not in self._param_warnings:
                    print(f"[DEBUG] Parameter '{key}' not supported by model, filtering out")
                    self._param_warnings.add(key)
        return filtered_batch

    def _distiller_needs_attention_outputs(self) -> bool:
        """Determine whether model forwards should request attentions/hidden states."""
        # Check if the underlying distiller (if any) needs attention outputs
        distiller = getattr(self.pipeline, 'distiller', None)
        if distiller is None:
            return False
        
        distiller_type = getattr(self, 'distiller_type', '')
        if distiller_type == 'attention':
            return True

        # Explicit opt-in from config for custom distillers.
        distil_cfg = self.config.get('distillation', {}) or {}
        if bool(distil_cfg.get('force_attention_outputs', False)):
            return True

        # Heuristic: distiller exposes attention-specific control knobs.
        attention_flags = (
            'use_attention_rollout',
            'use_cross_layer_flow',
            'use_dual_matching',
            'use_temporal_attention',
        )
        return any(hasattr(distiller, flag) for flag in attention_flags)
    
    def _compute_distillation_loss(
        self,
        student_outputs,
        teacher_outputs,
        batch: Dict[str, Any],
        feature_payload: Optional[Dict[str, Any]] = None
    ) -> tuple:
        """
        Compute distillation loss using the pipeline (unified path).
        
        All distillation now goes through the pipeline interface.
        Legacy distillers are auto-wrapped in SingleDistillerPipeline
        at init time, so this method always uses the pipeline path.
        
        Args:
            student_outputs: Student model outputs
            teacher_outputs: Teacher model outputs  
            batch: Input batch
            feature_payload: Ignored (kept for API compat)
        
        Returns:
            (loss, metrics_dict) tuple
        """
        batch.get('labels', None)
        pipeline_batch = {key: value for key, value in batch.items() if value is not None}
        
        # Build outputs dict for pipeline
        pipeline_outputs = {
            'teacher_outputs': teacher_outputs,
            'student_outputs': student_outputs,
            'batch': pipeline_batch,
        }
        
        # Compute loss through pipeline
        loss = self.pipeline.compute_loss(pipeline_outputs)
        
        # Get metrics from pipeline
        try:
            metrics = self.pipeline.get_metrics()
            metrics_dict = {
                'pipeline_metrics': metrics.to_dict(),
                'total_loss': loss.item() if hasattr(loss, 'item') else float(loss),
            }
            
            # Extract component losses if available
            if metrics.component_losses:
                metrics_dict.update(metrics.component_losses)
            
        except Exception as e:
            LOG.debug(f"Failed to get pipeline metrics: {e}")
            metrics_dict = {}
        
        return loss, metrics_dict

    def _build_forward_runtime_kwargs(self, model_params) -> Dict[str, Any]:
        """Build runtime kwargs compatible with the model forward signature."""
        kwargs: Dict[str, Any] = {}
        if not self._requires_attention_outputs:
            return kwargs

        if 'output_attentions' in model_params:
            kwargs['output_attentions'] = True
        if 'output_hidden_states' in model_params:
            kwargs['output_hidden_states'] = True
        if 'return_dict' in model_params:
            kwargs['return_dict'] = True
        return kwargs

    def _prepare_distiller_batch(self) -> None:
        """Allow distillers to clear transient caches before a new forward pass."""
        # Pipeline mode: check if the wrapped distiller has a prepare method
        distiller = getattr(self.pipeline, 'distiller', None)
        if distiller is None:
            return
        
        prepare_fn = getattr(distiller, 'prepare_for_forward_pass', None)
        if callable(prepare_fn):
            try:
                prepare_fn()
            except Exception:
                LOG.debug("Distiller prepare_for_forward_pass() failed", exc_info=True)

    def _collect_distiller_features(self) -> Dict[str, Dict[str, Any]]:
        """Collect optional hook/extractor features from the active distiller.
        
        In pipeline-first mode, pipelines handle feature extraction
        internally.  This method accesses the underlying distiller
        (if wrapped) for backward compatibility with code that
        calls it directly.
        """
        distiller = getattr(self.pipeline, 'distiller', None)
        if distiller is None:
            return {'student_features': {}, 'teacher_features': {}}

        def _coerce_feature(value: Any) -> Any:
            if isinstance(value, torch.Tensor):
                return value
            if isinstance(value, (tuple, list)):
                for item in value:
                    coerced = _coerce_feature(item)
                    if isinstance(coerced, torch.Tensor):
                        return coerced
                return None
            return value

        def _sanitize_feature_dict(features: Dict[str, Any]) -> Dict[str, Any]:
            sanitized: Dict[str, Any] = {}
            for key, value in features.items():
                coerced = _coerce_feature(value)
                if coerced is None:
                    continue
                sanitized[key] = coerced
            return sanitized

        teacher_features: Dict[str, Any] = {}
        student_features: Dict[str, Any] = {}

        teacher_hooks = getattr(distiller, 'teacher_hooks', None)
        if isinstance(teacher_hooks, dict) and teacher_hooks:
            teacher_features.update(teacher_hooks)

        student_hooks = getattr(distiller, 'student_hooks', None)
        if isinstance(student_hooks, dict) and student_hooks:
            student_features.update(student_hooks)

        teacher_extractor = getattr(distiller, 'teacher_extractor', None)
        if teacher_extractor is not None and hasattr(teacher_extractor, 'extract_attention_maps'):
            try:
                maps = teacher_extractor.extract_attention_maps()
                if isinstance(maps, dict) and maps:
                    teacher_features.update(maps)
            except Exception:
                LOG.debug("Failed collecting teacher extractor attention maps", exc_info=True)

        student_extractor = getattr(distiller, 'student_extractor', None)
        if student_extractor is not None and hasattr(student_extractor, 'extract_attention_maps'):
            try:
                maps = student_extractor.extract_attention_maps()
                if isinstance(maps, dict) and maps:
                    student_features.update(maps)
            except Exception:
                LOG.debug("Failed collecting student extractor attention maps", exc_info=True)

        return {
            'teacher_features': _sanitize_feature_dict(teacher_features),
            'student_features': _sanitize_feature_dict(student_features),
        }

    def _compute_supervised_fallback_loss(self, student_outputs: Any, labels: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Fallback loss when a distiller-specific compute_loss path fails."""
        if labels is None:
            return None

        logits = None
        if isinstance(student_outputs, dict):
            logits = student_outputs.get('logits')
        elif hasattr(student_outputs, 'logits'):
            logits = student_outputs.logits
        elif isinstance(student_outputs, tuple) and student_outputs:
            logits = student_outputs[0]
        else:
            logits = student_outputs

        if not isinstance(logits, torch.Tensor):
            return None

        # Causal LM token-level CE
        if logits.dim() == 3 and labels.dim() >= 2:
            ignore_index = int(self.config.get('distillation', {}).get('ignore_index', -100))
            shift_labels = bool(self.config.get('distillation', {}).get('shift_labels', True))
            if shift_labels:
                logits = logits[:, :-1, :].contiguous()
                labels = labels[:, 1:].contiguous()

            flat_logits = logits.view(-1, logits.size(-1))
            flat_labels = labels.reshape(-1)
            valid = flat_labels != ignore_index
            if valid.any():
                return F.cross_entropy(flat_logits[valid], flat_labels[valid])
            return logits.sum() * 0.0

        # Classification CE
        if logits.dim() >= 2:
            return F.cross_entropy(logits, labels)

        return None

    def train_epoch(self, dataloader):
        self.student.train()
        self.teacher.eval()
        total_loss = 0.0
        num_batches = 0
        accumulation_counter = 0  # For gradient accumulation
        grad_norm = 0.0  # Initialize for metrics streaming
        epoch_batch_train_losses: List[float] = []
        start_time_epoch = time.time()
        samples_processed = 0
        
        LOG.info(f"Starting training epoch with {len(dataloader)} batches")
        
        for batch_idx, batch in enumerate(dataloader):
            if not batch or not isinstance(batch, dict):
                print(f"[WARNING] Skipping empty or malformed batch at index {batch_idx}")
                continue
            try:
                batch = {k: v.to(self.device) for k, v in batch.items() if hasattr(v, 'to')}
            except Exception as e:
                print(f"[WARNING] Failed to move batch to device at index {batch_idx}: {e}")
                continue
            
            # Filter batch for each model's specific requirements
            teacher_batch = self._filter_batch_for_model(batch, self._teacher_forward_params)
            student_batch = self._filter_batch_for_model(batch, self._student_forward_params)
            teacher_runtime_kwargs = self._build_forward_runtime_kwargs(self._teacher_forward_params)
            student_runtime_kwargs = self._build_forward_runtime_kwargs(self._student_forward_params)
            self._prepare_distiller_batch()
            
            # ========== MIXED PRECISION TRAINING (AMP) ==========
            # Wrap forward pass in autocast for automatic mixed precision
            # Note: MPS doesn't support autocast, use CPU dtype
            amp_dtype = torch.bfloat16 if self.use_amp and self.device.type == 'cpu' else torch.float16
            with torch.no_grad():
                if self.use_amp and self.device.type != 'mps':
                    with torch.amp.autocast(device_type=self.device.type, dtype=amp_dtype):
                        try:
                            teacher_outputs = self.teacher(**teacher_batch, **teacher_runtime_kwargs)
                        except Exception as e:
                            print(f"[WARNING] Teacher forward pass failed at batch {batch_idx}: {e}")
                            continue
                else:
                    try:
                        teacher_outputs = self.teacher(**teacher_batch, **teacher_runtime_kwargs)
                    except Exception as e:
                        print(f"[WARNING] Teacher forward pass failed at batch {batch_idx}: {e}")
                        continue
            
            if self.use_amp and self.device.type != 'mps':
                with torch.amp.autocast(device_type=self.device.type, dtype=amp_dtype):
                    try:
                        student_outputs = self.student(**student_batch, **student_runtime_kwargs)
                    except Exception as e:
                        print(f"[WARNING] Student forward pass failed at batch {batch_idx}: {e}")
                        continue
                        
                    labels = batch.get('labels', None)
                    feature_payload = self._collect_distiller_features()
                    try:
                        # Use unified loss computation method
                        loss, metrics_dict = self._compute_distillation_loss(
                            student_outputs=student_outputs,
                            teacher_outputs=teacher_outputs,
                            batch=batch,
                            feature_payload=feature_payload
                        )
                        
                        # Scale loss for gradient accumulation
                        loss = loss / self.gradient_accumulation_steps
                            
                    except Exception as e:
                        print(f"[WARNING] Loss computation failed at batch {batch_idx}: {e}")
                        fallback_loss = self._compute_supervised_fallback_loss(student_outputs, labels)
                        if fallback_loss is None:
                            print(f"[WARNING] Skipping batch {batch_idx}: no valid fallback loss")
                            continue
                        loss = fallback_loss / self.gradient_accumulation_steps
                        print(f"[INFO] Using supervised fallback loss at batch {batch_idx}")
            else:
                try:
                    student_outputs = self.student(**student_batch, **student_runtime_kwargs)
                except Exception as e:
                    print(f"[WARNING] Student forward pass failed at batch {batch_idx}: {e}")
                    continue
                    
                labels = batch.get('labels', None)
                feature_payload = self._collect_distiller_features()
                try:
                    # Use unified loss computation method
                    loss, metrics_dict = self._compute_distillation_loss(
                        student_outputs=student_outputs,
                        teacher_outputs=teacher_outputs,
                        batch=batch,
                        feature_payload=feature_payload
                    )

                    # Scale loss for gradient accumulation
                    loss = loss / self.gradient_accumulation_steps
                except Exception as e:
                    print(f"[WARNING] Loss computation failed at batch {batch_idx}: {e}")
                    fallback_loss = self._compute_supervised_fallback_loss(student_outputs, labels)
                    if fallback_loss is None:
                        print(f"[WARNING] Skipping batch {batch_idx}: no valid fallback loss")
                        continue
                    loss = fallback_loss / self.gradient_accumulation_steps
                    print(f"[INFO] Using supervised fallback loss at batch {batch_idx}")
            
            # ========== GRADIENT ACCUMULATION ==========
            # Accumulate gradients over multiple batches
            if self.use_amp and self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            accumulation_counter += 1
            
            # Only step optimizer every N accumulation steps
            if accumulation_counter >= self.gradient_accumulation_steps:
                # Gradient management (clipping, centralization, monitoring)
                max_grad_norm = self.config['train'].get('max_grad_norm', 1.0)
                
                if self.use_amp and self.scaler:
                    # Unscale gradients before clipping
                    self.scaler.unscale_(self.optimizer)
                
                grad_norm = GradientManager.clip_gradients(
                    self.student,
                    max_norm=max_grad_norm,
                    norm_type=2.0
                )
                
                # Centralize gradients for distillation stability (optional)
                if self.config['train'].get('centralize_grads', False):
                    GradientManager.centralize_gradients(self.student)
                
                # Log gradient statistics if needed (every 100 batches)
                if batch_idx % 100 == 0 and grad_norm > 0:
                    grad_stats = GradientManager.get_gradient_stats(self.student)
                    if grad_stats['grad_norm'] > 10.0:  # Warn about potential gradient explosion
                        print(f"[WARN] Large gradient detected: norm={grad_stats['grad_norm']:.2f}, mean={grad_stats['grad_mean']:.4f}")
                
                # Step optimizer with AMP scaler
                if self.use_amp and self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                # Step scheduler per optimizer step for all schedulers
                # EXCEPT ReduceLROnPlateau (epoch-based)
                if self.scheduler is not None and hasattr(self.scheduler, 'step'):
                    scheduler_name = type(self.scheduler).__name__
                    if 'ReduceLROnPlateau' not in scheduler_name:
                        # Type safety: Check if step method requires metrics parameter
                        step_signature = inspect.signature(self.scheduler.step)
                        if 'metrics' in step_signature.parameters:
                            self.scheduler.step(metrics=None)  # type: ignore[call-arg]
                        else:
                            self.scheduler.step()  # type: ignore[call-arg]
                
                self.optimizer.zero_grad()
                accumulation_counter = 0
            
            # ========== LIVE METRICS STREAMING ==========
            # Broadcast real-time updates to UI via WebSocket
            if self.websocket_callback and batch_idx % self.update_frequency == 0:
                metrics_payload = {
                    'type': 'training_update',
                    'batch_idx': batch_idx,
                    'loss': loss.item() * self.gradient_accumulation_steps,  # Unscale for display
                    'grad_norm': grad_norm if accumulation_counter == 0 else None,
                    'lr': self.optimizer.param_groups[0]['lr']
                }
                try:
                    self.websocket_callback(metrics_payload)
                except Exception as e:
                    print(f"[WARNING] WebSocket callback failed: {e}")
            
            # ========== BATCH-LEVEL PROGRESS LOGGING ==========
            # Log detailed training progress every 10 batches
            if (batch_idx + 1) % 10 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                unscaled_loss = loss.item() * self.gradient_accumulation_steps
                grad_norm_str = f"{grad_norm:.3f}" if grad_norm > 0 else "N/A"
                LOG.info(
                    f"  Train Batch {batch_idx + 1}/{len(dataloader)}: "
                    f"Loss={unscaled_loss:.4f}, LR={current_lr:.2e}, "
                    f"GradNorm={grad_norm_str}"
                )
            
            total_loss += loss.item()
            # Track per-batch (scaled) loss for visualization
            try:
                epoch_batch_train_losses.append(float(loss.item()))
            except Exception:
                pass
            num_batches += 1
            # Batch size (approx) for throughput
            batch_size = 0
            if isinstance(batch, dict):
                labels_tensor = batch.get('labels')
                if labels_tensor is not None and hasattr(labels_tensor, 'shape'):
                    batch_size = labels_tensor.shape[0]
            samples_processed += batch_size
            # Detailed logging
            self._log_batch(
                phase='train',
                epoch=len(self.train_losses) + 1,  # next epoch index
                batch_idx=batch_idx,
                batches_total=len(dataloader),
                loss=loss.item() * self.gradient_accumulation_steps,
                scaled_loss=loss.item(),
                lr=self.optimizer.param_groups[0]['lr'],
                grad_norm=grad_norm if accumulation_counter == 0 else None,
                running_acc=None,  # Student training accuracy not computed here
                start_time_epoch=start_time_epoch,
                samples_processed=samples_processed,
                batch_size=batch_size,
                is_teacher=False
            )
        avg_loss = total_loss / max(num_batches, 1)
        self.train_losses.append(avg_loss)
        # Save epoch batch series
        self.batch_train_losses.append(epoch_batch_train_losses)
        # Generate micro-series plot for student training portion of this epoch
        try:
            micro_path = os.path.join(self.experiment_dir, f"student_epoch{len(self.train_losses)}_train_micro.png")
            plot_epoch_micro_series(
                title_prefix="Student",
                epoch_idx=len(self.train_losses),
                train_batch_losses=epoch_batch_train_losses,
                val_batch_losses=None,
                val_running_acc=None,
                save_path=micro_path,
            )
        except Exception as e:
            LOG.warning(f"Failed to plot student micro-series for epoch {len(self.train_losses)}: {e}")
        print(f"[TRAIN] Epoch training completed. Average Loss: {avg_loss:.4f}")
        return avg_loss

    def evaluate(self, dataloader, compute_extended=True) -> EvaluationReport:
        # Pre-evaluation Modality & Shape validation
        if len(dataloader) > 0:
            sample_batch = next(iter(dataloader))
            # Basic validation to ensure we don't crash deep in forward pass
            if "input_ids" not in sample_batch:
                LOG.warning("Evaluation batch missing 'input_ids'. This might crash the forward pass.")
                
        self.student.eval()
        self.teacher.eval()
        total_loss = 0.0
        num_batches = 0
        failed_batches = 0  # Track failed batches for validation
        all_preds = []
        all_labels = []
        epoch_val_losses: List[float] = []
        running_acc_series: List[float] = []
        running_correct = 0
        running_total = 0
        start_time_epoch = time.time()
        samples_processed = 0
        all_probabilities: List[np.ndarray] = []
        confidence_samples: List[float] = []
        latency_ms: List[float] = []
        peak_memory_bytes: List[int] = []
        is_token_task = False
        token_correct = 0
        token_total = 0
        lm_ignore_index = int(self.config.get('distillation', {}).get('ignore_index', -100))
        device_str = str(self.device)
        if isinstance(self.device, torch.device):
            device_str = self.device.type if self.device.index is None else f"{self.device.type}:{self.device.index}"
        if torch.cuda.is_available() and "cuda" in device_str:
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                LOG.debug("Unable to reset CUDA memory stats before evaluation", exc_info=True)
        
        # For extended metrics
        all_teacher_logits = []
        all_student_logits = []
        
        LOG.info(f"Starting evaluation with {len(dataloader)} batches")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if not batch or not isinstance(batch, dict):
                    print(f"[WARNING] Skipping empty or malformed batch at index {batch_idx} during evaluation")
                    failed_batches += 1
                    continue
                try:
                    batch = {k: v.to(self.device) for k, v in batch.items() if hasattr(v, 'to')}
                except Exception as e:
                    print(f"[WARNING] Failed to move batch to device at index {batch_idx} during evaluation: {e}")
                    failed_batches += 1
                    continue
                
                # Filter batch for each model's specific requirements
                teacher_batch = self._filter_batch_for_model(batch, self._teacher_forward_params)
                student_batch = self._filter_batch_for_model(batch, self._student_forward_params)
                teacher_runtime_kwargs = self._build_forward_runtime_kwargs(self._teacher_forward_params)
                student_runtime_kwargs = self._build_forward_runtime_kwargs(self._student_forward_params)
                self._prepare_distiller_batch()
                
                try:
                    teacher_outputs = self.teacher(**teacher_batch, **teacher_runtime_kwargs)
                except Exception as e:
                    print(f"[WARNING] Teacher forward pass failed at batch {batch_idx} during evaluation: {e}")
                    failed_batches += 1
                    continue
                forward_start = time.perf_counter()
                try:
                    student_outputs = self.student(**student_batch, **student_runtime_kwargs)
                except Exception as e:
                    print(f"[WARNING] Student forward pass failed at batch {batch_idx} during evaluation: {e}")
                    failed_batches += 1
                    continue
                batch_latency = (time.perf_counter() - forward_start) * 1000.0
                latency_ms.append(batch_latency)
                if torch.cuda.is_available() and "cuda" in device_str:
                    try:
                        peak_memory_bytes.append(torch.cuda.max_memory_allocated())
                    except Exception:
                        LOG.debug("Unable to read CUDA memory stats during evaluation", exc_info=True)
                labels = batch.get('labels', None)
                feature_payload = self._collect_distiller_features()
                try:
                    # Use unified loss computation method
                    loss, metrics_dict = self._compute_distillation_loss(
                        student_outputs=student_outputs,
                        teacher_outputs=teacher_outputs,
                        batch=batch,
                        feature_payload=feature_payload
                    )
                        
                except Exception as e:
                    print(f"[WARNING] Loss computation failed at batch {batch_idx} during evaluation: {e}")
                    loss = torch.tensor(0.0, device=self.device)
                    
                total_loss += loss.item()
                # Track per-batch validation loss
                try:
                    epoch_val_losses.append(float(loss.item()))
                except Exception:
                    pass
                num_batches += 1
                # Estimate batch size for throughput
                batch_size = 0
                labels_tensor = batch.get('labels') if isinstance(batch, dict) else None
                if labels_tensor is not None and hasattr(labels_tensor, 'shape'):
                    batch_size = labels_tensor.shape[0]
                samples_processed += batch_size
                # Detailed batch log with running accuracy if available
                current_running_acc = running_acc_series[-1] if running_acc_series else None
                self._log_batch(
                    phase='eval',
                    epoch=len(self.val_losses) + 1,
                    batch_idx=batch_idx,
                    batches_total=len(dataloader),
                    loss=loss.item(),
                    scaled_loss=loss.item(),
                    lr=self.optimizer.param_groups[0]['lr'],
                    grad_norm=None,
                    running_acc=current_running_acc,
                    start_time_epoch=start_time_epoch,
                    samples_processed=samples_processed,
                    batch_size=batch_size,
                    is_teacher=False
                )
                
                # Extract predictions - handle dict, object, or tensor
                if labels is not None:
                    # Extract logits for both teacher and student
                    teacher_logits = teacher_outputs.logits if hasattr(teacher_outputs, 'logits') else teacher_outputs.get('logits') if isinstance(teacher_outputs, dict) else teacher_outputs
                    student_logits = student_outputs.logits if hasattr(student_outputs, 'logits') else student_outputs.get('logits') if isinstance(student_outputs, dict) else student_outputs
                    
                    # Store for extended metrics
                    if compute_extended and teacher_logits is not None and student_logits is not None:
                        all_teacher_logits.append(teacher_logits.cpu())
                        all_student_logits.append(student_logits.cpu())
                    
                    if student_logits is not None and hasattr(student_logits, 'dim'):
                        if student_logits.dim() == 3 and hasattr(labels, 'dim') and labels.dim() >= 2:
                            is_token_task = True
                            token_preds = torch.argmax(student_logits, dim=-1)
                            token_labels = labels
                            if token_preds.size(1) > 1 and token_labels.size(1) > 1:
                                token_preds = token_preds[:, :-1]
                                token_labels = token_labels[:, 1:]

                            valid_mask = token_labels != lm_ignore_index
                            if valid_mask.any():
                                valid_preds = token_preds[valid_mask]
                                valid_labels = token_labels[valid_mask]
                                preds_list = valid_preds.detach().cpu().tolist()
                                labels_list = valid_labels.detach().cpu().tolist()
                                all_preds.extend(preds_list)
                                all_labels.extend(labels_list)
                                token_correct += int((valid_preds == valid_labels).sum().item())
                                token_total += int(valid_labels.numel())
                                running_acc = token_correct / max(token_total, 1)
                                running_acc_series.append(float(running_acc))
                        elif student_logits.dim() >= 2:
                            preds = torch.argmax(student_logits, dim=-1)
                            preds_list = preds.cpu().numpy().tolist()
                            labels_list = labels.cpu().numpy().tolist()
                            all_preds.extend(preds_list)
                            all_labels.extend(labels_list)
                        else:
                            preds = (student_logits > 0).long()
                            preds_list = preds.cpu().numpy().tolist()
                            labels_list = labels.cpu().numpy().tolist()
                            all_preds.extend(preds_list)
                            all_labels.extend(labels_list)

                        prob_tensor = None
                        if student_logits.dim() == 3:
                            # Token-level confidence snapshot for diagnostics.
                            prob_tensor = torch.softmax(student_logits, dim=-1)
                        elif student_logits.dim() >= 2:
                            prob_tensor = torch.softmax(student_logits, dim=-1)
                        else:
                            prob_tensor = torch.sigmoid(student_logits).unsqueeze(-1)

                        if prob_tensor is not None:
                            prob_cpu = prob_tensor.detach().cpu().numpy()
                            all_probabilities.append(prob_cpu)
                            try:
                                if prob_tensor.dim() >= 2:
                                    confidence_samples.extend(prob_tensor.max(dim=-1)[0].detach().cpu().tolist())
                                else:
                                    confidence_samples.extend(prob_tensor.detach().cpu().tolist())
                            except Exception:
                                pass

                        # Update running accuracy (classification path only)
                        if not is_token_task:
                            try:
                                batch_correct = int((preds.cpu() == labels.cpu()).sum().item())
                                batch_total = int(labels.numel())
                                running_correct += batch_correct
                                running_total += batch_total
                                running_acc = running_correct / max(running_total, 1)
                                running_acc_series.append(float(running_acc))
                            except Exception:
                                pass
                
                # Log progress every 10 batches
                if (batch_idx + 1) % 10 == 0:
                    LOG.info(f"  Eval Batch {batch_idx + 1}/{len(dataloader)}: Loss={loss.item():.4f}, Preds={len(all_preds)}, Labels={len(all_labels)}")
        
        # Validate evaluation results
        if num_batches == 0:
            raise RuntimeError(
                f"Evaluation failed: No valid batches processed! "
                f"Total batches in dataloader: {len(dataloader)}, "
                f"Failed batches: {failed_batches}"
            )
        
        max_allowed_failures = max(1, int(0.1 * len(dataloader)))  # Allow up to 10% failures
        if failed_batches > max_allowed_failures:
            LOG.warning(
                f"High failure rate during evaluation: {failed_batches}/{len(dataloader)} batches failed "
                f"({100 * failed_batches / len(dataloader):.1f}%). Max allowed: {max_allowed_failures}"
            )
        
        avg_loss = total_loss / num_batches  # Safe now since num_batches > 0
        self.val_losses.append(avg_loss)
        # Save epoch batch series
        self.batch_val_losses.append(epoch_val_losses)
        self.batch_val_running_acc.append(running_acc_series)
        # Generate micro-series plot combining val losses and running accuracy for this epoch
        try:
            micro_val_path = os.path.join(self.experiment_dir, f"student_epoch{len(self.val_losses)}_eval_micro.png")
            plot_epoch_micro_series(
                title_prefix="Student Eval",
                epoch_idx=len(self.val_losses),
                train_batch_losses=None,
                val_batch_losses=epoch_val_losses,
                val_running_acc=running_acc_series,
                save_path=micro_val_path,
            )
        except Exception as e:
            LOG.warning(f"Failed to plot student eval micro-series for epoch {len(self.val_losses)}: {e}")
        self.last_preds = all_preds
        self.last_labels = all_labels
        
        LOG.info(f"Evaluation complete: {num_batches} batches, {len(all_preds)} predictions, {len(all_labels)} labels")
        
        metrics = []
        pred_probs_np: Optional[np.ndarray] = None
        if all_probabilities:
            try:
                pred_probs_np = np.concatenate(all_probabilities, axis=0)
            except ValueError:
                LOG.warning("Failed to concatenate prediction probabilities; skipping probability-based metrics")
                pred_probs_np = None

        diagnostics: Dict[str, Any] = {}
        runtime_snapshot: Dict[str, Any] = {}
        calibration_payload: Optional[Dict[str, Any]] = None

        if is_token_task and token_total > 0:
            token_accuracy = token_correct / max(token_total, 1)
            perplexity = float(np.exp(min(avg_loss, 20.0)))
            lm_metrics = {
                'accuracy': token_accuracy,
                'token_accuracy': token_accuracy,
                'f1': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'perplexity': perplexity,
            }
            for key in self.metrics_history.keys():
                self.metrics_history[key].append(lm_metrics.get(key, 0.0))
            metrics = [lm_metrics]
            self.metrics_detail_history.append(lm_metrics)
            diagnostics = self._build_eval_diagnostics(
                lm_metrics,
                avg_loss,
                epoch_val_losses,
                confidence_samples,
                pred_probs_np,
            )
        elif all_labels and all_preds and len(all_labels) == len(all_preds):
            try:
                computed_metrics = compute_all_metrics(all_preds, all_labels, pred_probs=pred_probs_np)
                LOG.info(f"Computed metrics: {computed_metrics}")
                if not isinstance(computed_metrics, dict):
                    computed_metrics = {}
                for key in self.metrics_history.keys():
                    self.metrics_history[key].append(computed_metrics.get(key, 0))
                metrics = [computed_metrics]
                self.metrics_detail_history.append(computed_metrics)
                calibration_payload = computed_metrics.get('calibration') if isinstance(computed_metrics.get('calibration'), dict) else None
                diagnostics = self._build_eval_diagnostics(
                    computed_metrics,
                    avg_loss,
                    epoch_val_losses,
                    confidence_samples,
                    pred_probs_np,
                )
            except Exception as e:
                print(f"[WARNING] Metric computation failed during evaluation: {e}")
                LOG.error(f"Metric computation failed: {e}", exc_info=True)
        
        # Compute extended metrics if enabled
        extended_metrics = {}
        if compute_extended and all_teacher_logits and all_student_logits:
            try:
                teacher_logits_cat = torch.cat(all_teacher_logits, dim=0)
                student_logits_cat = torch.cat(all_student_logits, dim=0)
                
                temperature = self.config['distillation'].get('temperature', 2.0)
                try:
                    extended_metrics = compute_extended_metrics(
                        teacher_logits_cat, 
                        student_logits_cat,
                        temperature=temperature
                    )
                except Exception as e:
                    LOG.warning(f"Extended metrics computation failed: {e}")
                    extended_metrics = {'dei': 0.0, 'cas': 0.0}
                
                # Track extended metrics
                for key in self.extended_metrics_history.keys():
                    if key in extended_metrics:
                        self.extended_metrics_history[key].append(extended_metrics[key])
                
                print(f"[EXTENDED] KL: {extended_metrics['kl_divergence']:.4f}, "
                      f"Agreement: {extended_metrics['prediction_agreement']:.2%}")
            except Exception as e:
                print(f"[WARNING] Extended metrics computation failed: {e}")
        
        if latency_ms:
            lat_arr = np.asarray(latency_ms, dtype=np.float64)
            runtime_snapshot = {
                'mean_ms': float(lat_arr.mean()),
                'median_ms': float(np.median(lat_arr)),
                'p95_ms': float(np.percentile(lat_arr, 95)),
                'p99_ms': float(np.percentile(lat_arr, 99)),
                'max_ms': float(lat_arr.max()),
                'min_ms': float(lat_arr.min()),
                'batches': len(lat_arr),
            }
            total_time_s = lat_arr.sum() / 1000.0
            if total_time_s > 0 and samples_processed > 0:
                runtime_snapshot['throughput_samples_per_s'] = float(samples_processed / total_time_s)
            if peak_memory_bytes:
                peak_mem = np.asarray(peak_memory_bytes, dtype=np.float64)
                runtime_snapshot['peak_memory_mb'] = float(peak_mem.max() / (1024 ** 2))
        if runtime_snapshot:
            self.eval_runtime_history.append(runtime_snapshot)
        if diagnostics:
            self.eval_diagnostics_history.append(diagnostics)
        if calibration_payload:
            self.eval_calibration_history.append(calibration_payload)

        print(f"[EVAL] Evaluation completed. Average Loss: {avg_loss:.4f}, Metrics: {metrics if metrics else 'N/A'}")
        if diagnostics.get('warnings'):
            LOG.info(f"Evaluation diagnostics flagged: {diagnostics['warnings']}")

        details = {
            'metrics': metrics[0] if metrics else {},
            'diagnostics': diagnostics,
            'runtime': runtime_snapshot,
            'calibration': calibration_payload,
        }

        # Build the evaluation report
        report = EvaluationReport(
            loss=avg_loss,
            metrics=(metrics[0] if metrics else {}),
            diagnostics=diagnostics,
            runtime=runtime_snapshot or None,
            calibration=calibration_payload,
            explainability=None,
            modality=self.config.get('data', {}).get('modality', 'text'),
            model_name=self.student.__class__.__name__,
            task_type=self.config.get('data', {}).get('task_type', 'classification'),
            distillation_metrics=extended_metrics or None,
            metadata={'details': details},
        )
        return report

    def _build_eval_diagnostics(
        self,
        metrics: Dict[str, Any],
        avg_loss: Optional[float],
        batch_losses: List[float],
        confidence_samples: List[float],
        probabilities: Optional[np.ndarray],
    ) -> Dict[str, Any]:
        try:
            return build_eval_diagnostics(
                metrics=metrics,
                avg_loss=avg_loss,
                batch_losses=batch_losses,
                confidence_samples=confidence_samples,
                probabilities=probabilities,
            )
        except Exception:
            LOG.debug("Failed to compute evaluation diagnostics", exc_info=True)
            return {'warnings': ['diagnostics_failed']}

    def fit(self, train_loader, val_loader):
        # ========== DATA VALIDATION ==========
        # Check for data leakage and quality issues before training
        LOG.info("="*80)
        LOG.info("PRE-TRAINING DATA VALIDATION")
        LOG.info("="*80)
        
        try:
            # Extract samples from dataloaders for validation
            train_samples = []
            val_samples = []
            
            # Collect samples (limit to 1000 for performance)
            max_check_samples = 1000
            for i, batch in enumerate(train_loader):
                if i >= max_check_samples // train_loader.batch_size:
                    break
                # Assuming batch has 'input_ids' and 'labels'
                if 'input_ids' in batch and 'labels' in batch:
                    input_ids = batch['input_ids'].cpu().numpy()
                    labels = batch['labels'].cpu().numpy()
                    # Decode texts for overlap checking
                    for j in range(len(input_ids)):
                        text = self.tokenizer.decode(input_ids[j], skip_special_tokens=True)
                        train_samples.append({'text': text, 'label': int(labels[j])})
            
            for i, batch in enumerate(val_loader):
                if i >= max_check_samples // val_loader.batch_size:
                    break
                if 'input_ids' in batch and 'labels' in batch:
                    input_ids = batch['input_ids'].cpu().numpy()
                    labels = batch['labels'].cpu().numpy()
                    for j in range(len(input_ids)):
                        text = self.tokenizer.decode(input_ids[j], skip_special_tokens=True)
                        val_samples.append({'text': text, 'label': int(labels[j])})
            
            # Validate split
            if train_samples and val_samples:
                validation_results = DataValidator.validate_dataset_split(
                    train_samples,
                    val_samples,
                    text_key='text',
                    label_key='label'
                )
                
                # Save validation report
                validation_path = Path(self.experiment_dir) / 'data_validation_report.json'
                DataValidator.save_validation_report(validation_results, validation_path)
                
                # Halt if critical errors found
                if not validation_results['validation_passed']:
                    LOG.error("="*80)
                    LOG.error("DATA VALIDATION FAILED!")
                    LOG.error("="*80)
                    for error in validation_results['errors']:
                        LOG.error(f"  [FAIL] {error}")
                    LOG.error("\nPlease fix data issues before training.")
                    LOG.error("See report: " + str(validation_path))
                    LOG.error("="*80)
                    
                    fail_policy = str(
                        self.config.get('train', {}).get('data_validation_fail_policy', 'warn')
                    ).lower()
                    if fail_policy == 'abort':
                        raise ValueError("Training aborted due to data validation failure")
                    LOG.warning(
                        "Continuing despite data validation failure because "
                        "train.data_validation_fail_policy=%s",
                        fail_policy,
                    )
        
        except Exception as e:
            LOG.warning(f"Data validation check failed: {e}")
            LOG.warning("Continuing with training...")
        
        LOG.info("="*80)
        # ========== END DATA VALIDATION ==========
        
        # Initialize scheduler now that we know steps_per_epoch
        if self.scheduler is None:
            steps_per_epoch = len(train_loader)
            num_epochs = self.config['train'].get('epochs', 10)
            total_steps = steps_per_epoch * num_epochs
            
            # FIX: Log scheduler initialization details
            warmup_steps = self.config['train'].get('warmup_steps', 0)
            LOG.info("Initializing scheduler with:")
            LOG.info(f"  Steps per epoch: {steps_per_epoch}")
            LOG.info(f"  Total epochs: {num_epochs}")
            LOG.info(f"  Total training steps: {total_steps}")
            LOG.info(f"  Warmup steps: {warmup_steps}")
            LOG.info(f"  Effective cosine steps: {total_steps - warmup_steps}")
            
            scheduler_factory = SchedulerFactory(self.optimizer, self.config['train'])
            self.scheduler = scheduler_factory.get_scheduler(num_training_steps=total_steps)
            print(f"[INFO] Scheduler initialized: {type(self.scheduler).__name__}")
            
            # FIX: Log initial LR to verify scheduler is working
            initial_lr = self.optimizer.param_groups[0]['lr']
            LOG.info(f"  Initial learning rate: {initial_lr:.2e}")
        
        # Resume from checkpoint if requested
        resume_from = self.config.get('train', {}).get('resume_from_checkpoint')
        if resume_from:
            self.resume_from_checkpoint(resume_from)
            
        self._init_mitigation_context(train_loader)

        # Optional: Train teacher first if configured
        if self.should_train_teacher:
            print("\n" + "=" * 70)
            print("PHASE 2.5: Fine-tuning Teacher Model")
            print("=" * 70 + "\n")
            print(f"[INFO] Training teacher for {self.teacher_epochs} epochs before distillation...")
            self._train_teacher(train_loader, val_loader)
            print("[INFO] Teacher training completed. Starting distillation...\n")
        
        metrics_history = []
        epochs = self.config['train']['epochs']
        start_epoch = int(max(0, self.resume_epoch))
        if start_epoch >= epochs:
            print(f"[INFO] Resume epoch {start_epoch} is >= configured epochs ({epochs}); skipping training loop.")
        else:
            print(f"[INFO] Training started for {epochs} epochs (starting at epoch {start_epoch + 1}).")
        for epoch in range(start_epoch, epochs):
            print(f"[INFO] Starting epoch {epoch+1}/{epochs}")
            train_loss = self.train_epoch(train_loader)
            val_report = self.evaluate(val_loader, compute_extended=True)
            val_loss = val_report.loss if val_report.loss is not None else float(val_report.metrics.get('loss', 0.0))
            val_metrics = [val_report.metrics]
            extended = val_report.distillation_metrics
            val_details = {
                'metrics': val_report.metrics,
                'diagnostics': val_report.diagnostics,
                'runtime': val_report.runtime or {},
                'calibration': val_report.calibration,
            }
            
            # Post-evaluation hooks for pipeline telemetry
            if hasattr(self, 'pipeline') and hasattr(self.pipeline, 'on_eval_complete'):
                try:
                    self.pipeline.on_eval_complete(val_report)
                except Exception as e:
                    LOG.warning(f"Pipeline post-evaluate hook failed: {e}")
                    
            self.final_report = val_report
            if not isinstance(val_metrics, list):
                val_metrics = []
            val_metrics_dict = val_metrics[0] if val_metrics else {}
            epoch_record = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
            }
            epoch_record.update({k: v for k, v in val_metrics_dict.items() if isinstance(v, (int, float, np.floating))})

            if val_details.get('diagnostics'):
                diag = val_details['diagnostics']
                if diag.get('warnings'):
                    print(f"[DIAGNOSTICS] Evaluation warnings: {diag['warnings']}")
            if val_details.get('runtime'):
                runtime_snapshot = val_details['runtime']
                if runtime_snapshot:
                    print(
                        "[RUNTIME] Eval latency mean={:.2f}ms throughput={:.2f} samples/s".format(
                            runtime_snapshot.get('mean_ms', 0.0),
                            runtime_snapshot.get('throughput_samples_per_s', 0.0),
                        )
                    )
            if val_details.get('calibration'):
                cal = val_details['calibration']
                if cal and isinstance(cal, dict):
                    brier_score = cal.get('brier_score')
                    if brier_score is not None:
                        print(f"[CALIBRATION] Brier score: {float(brier_score):.4f}")

            numeric_metrics = {
                k: float(v)
                for k, v in (val_details.get('metrics') or {}).items()
                if isinstance(v, (int, float, np.floating))
            }
            epoch_record.update(numeric_metrics)
            metrics_history.append(epoch_record)

            print(f"[INFO] Epoch {epoch+1} summary: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Metrics={val_metrics_dict}")

            # Step scheduler only for epoch-based schedulers (ReduceLROnPlateau)
            if self.scheduler is not None:
                scheduler_name = type(self.scheduler).__name__
                if 'ReduceLROnPlateau' in scheduler_name:
                    # ReduceLROnPlateau needs a metric
                    metric_value = val_metrics_dict.get('accuracy', val_loss)
                    self.scheduler.step(metric_value)
            
            # Adaptive LR tuning based on extended metrics (DEI/CAS)
            if hasattr(self, 'adaptive_opt') and extended:
                actions = self.adaptive_opt.auto_tune(extended, epoch=epoch+1)
                if actions:
                    print(f"[ADAPTIVE] LR tuning actions: {', '.join(actions)}")
            
            # Log current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"[INFO] Current learning rate: {current_lr:.6e}")

            early_stop_triggered = False
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = copy.deepcopy(self.student.state_dict())
                self.no_improve_epochs = 0
                print(f"[INFO] Validation loss improved to {val_loss:.4f}. Saving best model state.")
            else:
                self.no_improve_epochs += 1
                print(f"[INFO] No improvement in validation loss for {self.no_improve_epochs} epoch(s).")
                if self.no_improve_epochs >= self.early_stop_patience:
                    print(f"[INFO] Early stopping triggered after {self.no_improve_epochs} epochs without improvement.")
                    early_stop_triggered = True

            guard_triggered = self._maybe_trigger_overfit_guard(epoch + 1, train_loader=train_loader)

            if early_stop_triggered:
                break
            if guard_triggered:
                break
                
            # Per-epoch checkpointing
            save_epochs = self.config.get('train', {}).get('save_epochs', 0)
            if save_epochs > 0 and (epoch + 1) % save_epochs == 0:
                ckpt_dir = os.path.join(self.experiment_dir, f'checkpoint_epoch_{epoch+1}')
                ModelSaver.save_training_run(
                    self.student,
                    self.tokenizer,
                    ckpt_dir,
                    config=self.config,
                    metrics_history=self.metrics_history,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    scaler=self.scaler,
                    epoch=epoch + 1,
                    global_step=(epoch + 1) * max(1, len(train_loader)),
                    best_metric=(val_metrics_dict.get('accuracy') if isinstance(val_metrics_dict, dict) else None),
                    evaluation_report=val_report,
                )
                print(f"[INFO] Saved epoch {epoch+1} checkpoint to {ckpt_dir}")

        # Final summary
        self._summarize_training(metrics_history)
        
        # Analyze training health (overfitting/underfitting)
        if len(self.train_losses) >= 2 and len(self.val_losses) >= 2:
            training_health = OverfitUnderfitDetector.get_training_health_summary(
                self.train_losses,
                self.val_losses,
                self.metrics_history
            )
            print(training_health)
            
            # Save health analysis
            try:
                health_path = os.path.join(self.experiment_dir, 'training_health.json')
                health_analysis = OverfitUnderfitDetector.analyze_training_curves(
                    self.train_losses,
                    self.val_losses,
                    self.metrics_history.get('accuracy', []),
                    self.metrics_history.get('accuracy', []),  # Will be split by train/val in future
                    metric_name='accuracy'
                )
                events: List[Dict[str, Any]] = []
                for event in self.overfit_guard_state.get('events', []):
                    sanitized = {k: v for k, v in event.items() if k != 'analysis'}
                    if isinstance(event.get('analysis'), dict):
                        analysis_summary = {
                            'status': event['analysis'].get('status'),
                            'confidence': event['analysis'].get('confidence'),
                            'loss_gap_pct': event['analysis'].get('loss_gap_pct')
                        }
                        sanitized['analysis'] = analysis_summary
                    events.append(sanitized)

                health_analysis['overfit_guard'] = {
                    'enabled': self.overfit_guard_config['enabled'],
                    'mode': self.overfit_guard_config['mode'],
                    'triggered': self.overfit_guard_state.get('triggered', False),
                    'epoch': self.overfit_guard_state.get('epoch'),
                    'status': self.overfit_guard_state.get('status'),
                    'confidence': self.overfit_guard_state.get('confidence'),
                    'loss_gap_pct': self.overfit_guard_state.get('loss_gap_pct'),
                    'action': self.overfit_guard_state.get('action'),
                    'events': events,
                    'mitigation': {
                        'enabled': self.overfit_mitigation_config.get('enabled', True),
                        'history': self.overfit_mitigation_state.get('history', []),
                        'counts': self.overfit_mitigation_state.get('counts', {})
                    }
                }
                with open(health_path, 'w') as f:
                    json.dump(health_analysis, f, indent=2)
                LOG.info(f"Training health analysis saved to {health_path}")
            except Exception as e:
                LOG.warning(f"Failed to save training health analysis: {e}")

        # Restore best model and save
        if self.best_model_state:
            self.student.load_state_dict(self.best_model_state)
            print('[INFO] Restored best model before saving.')
            
            # Save student model
            student_save_dir = os.path.join(self.experiment_dir, 'student_model')
            ModelSaver.save_training_run(
                self.student,
                self.tokenizer,
                student_save_dir,
                config=self.config,
                metrics_history=self.metrics_history,
                evaluation_report=getattr(self, 'final_report', None),
            )
            print(f'[INFO] Student model and tokenizer saved to {student_save_dir}')
            
            # Save teacher model for comparison
            teacher_save_dir = os.path.join(self.experiment_dir, 'teacher_model')
            ModelSaver.save_training_run(
                self.teacher,
                self.tokenizer,
                teacher_save_dir,
                config=self.config,
            )
            print(f'[INFO] Teacher model and tokenizer saved to {teacher_save_dir}')
        else:
            print('[WARNING] No best model state found to restore.')

        # Plot once at the end safely
        agg_metrics = {k: self.metrics_history[k] for k in self.metrics_history.keys()}
        save_path = os.path.join(self.experiment_dir, 'training_curves.png')
        try:
            plot_training_curves(
                self.train_losses,
                self.val_losses,
                agg_metrics,
                save_path,
                lr_history=None,
                batch_train_losses=self.batch_train_losses,
                batch_val_losses=self.batch_val_losses,
                batch_val_running_acc=self.batch_val_running_acc,
            )
            print(f"[INFO] Training curves saved to {save_path}")
        except Exception as e:
            print(f"[WARNING] Failed to plot training curves: {e}")

        # Confusion matrices for teacher and student (if predictions available)
        try:
            # Student confusion matrix
            if self.last_preds and self.last_labels and len(self.last_preds) == len(self.last_labels):
                student_cm_dir = os.path.join(self.experiment_dir, 'student_confusion')
                os.makedirs(student_cm_dir, exist_ok=True)
                student_metrics = compute_all_metrics(self.last_preds, self.last_labels)
                plot_metrics([student_metrics], student_cm_dir)
                print(f"[CONFUSION] Student confusion matrix saved to {student_cm_dir}")
            # Teacher confusion matrix
            if getattr(self, 'teacher_last_preds', None) and getattr(self, 'teacher_last_labels', None) and len(self.teacher_last_preds) == len(self.teacher_last_labels):
                teacher_cm_dir = os.path.join(self.experiment_dir, 'teacher_confusion')
                os.makedirs(teacher_cm_dir, exist_ok=True)
                teacher_metrics = compute_all_metrics(self.teacher_last_preds, self.teacher_last_labels)
                plot_metrics([teacher_metrics], teacher_cm_dir)
                print(f"[CONFUSION] Teacher confusion matrix saved to {teacher_cm_dir}")
        except Exception as e:
            print(f"[WARNING] Failed to generate confusion matrices: {e}")

        if self.last_preds and self.last_labels and len(self.last_preds) == len(self.last_labels):
            try:
                from zynthe.evaluation.metrics import compute_all_metrics as _cm_all  # type: ignore
                from zynthe.evaluation.metrics import plot_metrics as _plot_metrics  # type: ignore
                final_metrics = _cm_all(self.last_preds, self.last_labels)
                if isinstance(final_metrics, dict):
                    _plot_metrics([final_metrics], self.experiment_dir)
                    print(f"[INFO] Final metrics plotted and saved in {self.experiment_dir}")
            except Exception as e:
                print(f"[WARNING] Failed to plot final metrics: {e}")
        else:
            print("[INFO] Skipping final metrics plotting due to missing or mismatched predictions and labels.")
        
        # Save extended metrics history
        try:
            extended_metrics_path = os.path.join(self.experiment_dir, 'extended_metrics.json')
            with open(extended_metrics_path, 'w') as f:
                json.dump(self.extended_metrics_history, f, indent=2)
            print(f"[INFO] Extended metrics saved to {extended_metrics_path}")
        except Exception as e:
            print(f"[WARNING] Failed to save extended metrics: {e}")

        # Evaluation dashboard is a core run artifact, not only a teacher-comparison artifact.
        if hasattr(self, 'final_report') and self.final_report:
            try:
                report_metadata = dict(getattr(self.final_report, 'metadata', {}) or {})
                report_metadata.update({
                    'train_losses': list(self.train_losses),
                    'val_losses': list(self.val_losses),
                    'metrics_history': {k: list(v) for k, v in self.metrics_history.items()},
                    'extended_metrics_history': {
                        k: list(v) for k, v in self.extended_metrics_history.items()
                    },
                })
                self.final_report.metadata = report_metadata

                dash_path = os.path.join(self.experiment_dir, 'evaluation_dashboard.png')
                plot_evaluation_dashboard(self.final_report, dash_path)

                extended_plot_path = os.path.join(self.experiment_dir, 'extended_metrics.png')
                plot_extended_metrics(self.final_report, extended_plot_path)
            except Exception as e:
                print(f"[WARNING] Failed to generate evaluation dashboard artifacts: {e}")

        # Teacher vs Student comparison plot (if enabled and data exists)
        if self.enable_comparison_plot and self.teacher_epoch_losses and self.train_losses:
            try:
                # Legacy comparison plot
                comparison_path = os.path.join(self.experiment_dir, 'teacher_student_comparison.png')
                plot_teacher_student_comparison(
                    student_train_losses=self.train_losses,
                    student_val_losses=self.val_losses,
                    teacher_train_losses=self.teacher_epoch_losses,
                    teacher_val_losses=self.teacher_epoch_val_losses,
                    student_metrics={k: v for k, v in self.metrics_history.items()},
                    teacher_metrics={k: v for k, v in self.teacher_metrics_history.items()},
                    save_path=comparison_path
                )
                
                # Distillation gap is only meaningful when teacher metrics are available.
                if hasattr(self, 'final_report') and self.final_report:
                    gap_path = os.path.join(self.experiment_dir, 'distillation_gap.png')
                    plot_distillation_gap(
                        teacher_metrics={k: v[-1] if v else 0.0 for k, v in self.teacher_metrics_history.items()},
                        student_metrics={k: v[-1] if v else 0.0 for k, v in self.metrics_history.items()},
                        save_path=gap_path
                    )
            except Exception as e:
                print(f"[WARNING] Failed to plot teacher-student comparison: {e}")

    def _summarize_training(self, metrics_history):
        print('\n[SUMMARY] Training Summary:')
        for record in metrics_history:
            if not isinstance(record, dict):
                continue
            epoch = record.get('epoch', '?')
            train_loss = record.get('train_loss', 0)
            val_loss = record.get('val_loss', 0)
            accuracy = record.get('accuracy', 0)
            f1 = record.get('f1', 0)
            precision = record.get('precision', 0)
            recall = record.get('recall', 0)
            print(f'[SUMMARY] Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, '
                  f'Accuracy={accuracy:.4f}, F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}')
        # No plotting here to avoid repeated visualizations

    def _train_teacher(self, train_loader, val_loader):
        """Train/fine-tune the teacher model on the task before distillation."""
        def _extract_logits(output: Any) -> Optional[torch.Tensor]:
            if isinstance(output, dict):
                return output.get('logits')
            logits_attr = getattr(output, 'logits', None)
            if logits_attr is not None:
                return logits_attr
            if isinstance(output, tuple) and len(output) > 0 and torch.is_tensor(output[0]):
                return output[0]
            return None

        def _teacher_loss(output: Any, labels: Optional[torch.Tensor]) -> torch.Tensor:
            # Preferred path when model returns native loss
            if hasattr(output, 'loss') and getattr(output, 'loss') is not None:
                return output.loss
            if isinstance(output, dict) and output.get('loss') is not None:
                return output['loss']

            logits = _extract_logits(output)
            if logits is None or labels is None:
                raise KeyError("loss")

            # Causal-LM fallback
            if logits.dim() == 3 and labels.dim() >= 2:
                ignore_index = int(self.config.get('distillation', {}).get('ignore_index', -100))
                shift_labels = bool(self.config.get('distillation', {}).get('shift_labels', True))
                if shift_labels:
                    logits = logits[:, :-1, :].contiguous()
                    labels_local = labels[:, 1:].contiguous()
                else:
                    labels_local = labels

                flat_logits = logits.view(-1, logits.size(-1))
                flat_labels = labels_local.reshape(-1)
                valid = flat_labels != ignore_index
                if valid.any():
                    return F.cross_entropy(flat_logits[valid], flat_labels[valid])
                return logits.sum() * 0.0

            # Classification fallback
            return F.cross_entropy(logits, labels)

        # Ensure teacher is in training mode and gradients are enabled
        self.teacher.train()
        # Explicitly enable gradients for teacher parameters
        for param in self.teacher.parameters():
            param.requires_grad = True
        
        best_teacher_loss = float('inf')
        best_teacher_state = None
        self.teacher_batch_train_losses = []  # List[List[float]]
        self.teacher_batch_val_losses = []    # List[List[float]]
        self.teacher_batch_val_running_acc = []  # List[List[float]]
        self.teacher_epoch_losses = []
        self.teacher_epoch_val_losses = []
        self.teacher_last_preds = []
        self.teacher_last_labels = []
        
        for epoch in range(self.teacher_epochs):
            print(f"[TEACHER] Epoch {epoch+1}/{self.teacher_epochs}")
            
            # Training
            total_loss = 0.0
            num_batches = 0
            epoch_train_batch_losses = []
            running_train_correct = 0
            running_train_total = 0
            running_train_acc_series = []
            start_time_epoch = time.time()
            samples_processed = 0
            for batch_idx, batch in enumerate(train_loader):
                if not batch or not isinstance(batch, dict):
                    continue
                
                batch = {k: v.to(self.device) for k, v in batch.items() if hasattr(v, 'to')}
                teacher_batch = self._filter_batch_for_model(batch, self._teacher_forward_params)
                
                self.teacher_optimizer.zero_grad()
                outputs = self.teacher(**teacher_batch)
                loss = _teacher_loss(outputs, batch.get('labels'))
                
                # Ensure loss requires grad
                if not loss.requires_grad:
                    LOG.warning(f"Teacher loss at batch {batch_idx} does not require grad, skipping backward")
                    continue
                
                loss.backward()
                self.teacher_optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                try:
                    epoch_train_batch_losses.append(float(loss.item()))
                except Exception:
                    pass
                # Compute training accuracy for this batch if labels available
                try:
                    logits = outputs.logits if hasattr(outputs, 'logits') else outputs['logits']
                    labels = teacher_batch.get('labels')
                    if labels is not None and logits is not None and hasattr(logits, 'dim') and logits.dim() >= 2:
                        preds = torch.argmax(logits, dim=-1)
                        running_train_correct += int((preds == labels).sum().item())
                        running_train_total += int(labels.numel())
                        running_train_acc_series.append(running_train_correct / max(running_train_total, 1))
                except Exception:
                    pass
                # Throughput + detailed logging
                batch_size = 0
                labels_tensor = batch.get('labels') if isinstance(batch, dict) else None
                if labels_tensor is not None and hasattr(labels_tensor, 'shape'):
                    batch_size = labels_tensor.shape[0]
                samples_processed += batch_size
                running_acc_val = running_train_acc_series[-1] if running_train_acc_series else None
                self._log_batch(
                    phase='teacher_train',
                    epoch=epoch + 1,
                    batch_idx=batch_idx,
                    batches_total=len(train_loader),
                    loss=loss.item(),
                    scaled_loss=loss.item(),
                    lr=self.teacher_optimizer.param_groups[0]['lr'],
                    grad_norm=None,
                    running_acc=running_acc_val,
                    start_time_epoch=start_time_epoch,
                    samples_processed=samples_processed,
                    batch_size=batch_size,
                    is_teacher=True
                )
            
            avg_train_loss = total_loss / max(num_batches, 1)
            self.teacher_epoch_losses.append(avg_train_loss)
            self.teacher_batch_train_losses.append(epoch_train_batch_losses)
            # Micro-series plot for teacher training epoch
            try:
                teacher_train_micro = os.path.join(self.experiment_dir, f"teacher_epoch{epoch+1}_train_micro.png")
                plot_epoch_micro_series(
                    title_prefix="Teacher Train",
                    epoch_idx=epoch+1,
                    train_batch_losses=epoch_train_batch_losses,
                    val_batch_losses=None,
                    val_running_acc=running_train_acc_series if running_train_acc_series else None,
                    save_path=teacher_train_micro,
                )
            except Exception as e:
                LOG.warning(f"Failed to plot teacher train micro-series for epoch {epoch+1}: {e}")
            
            # Validation
            self.teacher.eval()
            val_loss = 0.0
            val_batches = 0
            all_preds = []
            all_labels = []
            epoch_val_batch_losses = []
            running_val_correct = 0
            running_val_total = 0
            running_val_acc_series = []
            start_time_val = time.time()
            samples_processed_val = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    if not batch or not isinstance(batch, dict):
                        continue
                    
                    batch = {k: v.to(self.device) for k, v in batch.items() if hasattr(v, 'to')}
                    teacher_batch = self._filter_batch_for_model(batch, self._teacher_forward_params)
                    
                    outputs = self.teacher(**teacher_batch)
                    loss = _teacher_loss(outputs, batch.get('labels'))
                    val_loss += loss.item()
                    val_batches += 1
                    try:
                        epoch_val_batch_losses.append(float(loss.item()))
                    except Exception:
                        pass
                    
                    # Get predictions
                    logits = outputs.logits if hasattr(outputs, 'logits') else outputs['logits']
                    preds = torch.argmax(logits, dim=-1)
                    labels = batch.get('labels')
                    
                    if labels is not None:
                        preds_list = preds.cpu().numpy().tolist()
                        labels_list = labels.cpu().numpy().tolist()
                        all_preds.extend(preds_list)
                        all_labels.extend(labels_list)
                        try:
                            batch_correct = int((preds == labels).sum().item())
                            batch_total = int(labels.numel())
                            running_val_correct += batch_correct
                            running_val_total += batch_total
                            running_val_acc_series.append(running_val_correct / max(running_val_total, 1))
                        except Exception:
                            pass
                    # Detailed eval batch logging for teacher
                    batch_size = 0
                    labels_tensor = batch.get('labels') if isinstance(batch, dict) else None
                    if labels_tensor is not None and hasattr(labels_tensor, 'shape'):
                        batch_size = labels_tensor.shape[0]
                    samples_processed_val += batch_size
                    running_acc_val = running_val_acc_series[-1] if running_val_acc_series else None
                    self._log_batch(
                        phase='teacher_eval',
                        epoch=epoch + 1,
                        batch_idx=len(epoch_val_batch_losses)-1,
                        batches_total=len(val_loader),
                        loss=loss.item(),
                        scaled_loss=loss.item(),
                        lr=self.teacher_optimizer.param_groups[0]['lr'],
                        grad_norm=None,
                        running_acc=running_acc_val,
                        start_time_epoch=start_time_val,
                        samples_processed=samples_processed_val,
                        batch_size=batch_size,
                        is_teacher=True
                    )
            
            avg_val_loss = val_loss / max(val_batches, 1)
            self.teacher_epoch_val_losses.append(avg_val_loss)
            self.teacher_batch_val_losses.append(epoch_val_batch_losses)
            self.teacher_batch_val_running_acc.append(running_val_acc_series)
            # Micro-series plot for teacher evaluation epoch
            try:
                teacher_eval_micro = os.path.join(self.experiment_dir, f"teacher_epoch{epoch+1}_eval_micro.png")
                plot_epoch_micro_series(
                    title_prefix="Teacher Eval",
                    epoch_idx=epoch+1,
                    train_batch_losses=None,
                    val_batch_losses=epoch_val_batch_losses,
                    val_running_acc=running_val_acc_series if running_val_acc_series else None,
                    save_path=teacher_eval_micro,
                )
            except Exception as e:
                LOG.warning(f"Failed to plot teacher eval micro-series for epoch {epoch+1}: {e}")
            
            # Compute metrics
            teacher_metrics = {}
            if all_preds and all_labels:
                # For causal-LM, predictions/labels are token grids (multiclass-multioutput)
                # and sklearn classification helpers will fail. Compute token accuracy directly.
                first_label = all_labels[0]
                is_token_grid = isinstance(first_label, (list, tuple, np.ndarray))

                if is_token_grid:
                    flat_preds: List[int] = []
                    flat_labels: List[int] = []
                    ignore_index = int(self.config.get('distillation', {}).get('ignore_index', -100))

                    for pred_row, label_row in zip(all_preds, all_labels):
                        if not isinstance(pred_row, (list, tuple, np.ndarray)):
                            pred_row = [pred_row]
                        if not isinstance(label_row, (list, tuple, np.ndarray)):
                            label_row = [label_row]
                        for p, l in zip(pred_row, label_row):
                            try:
                                li = int(l)
                            except Exception:
                                continue
                            if li == ignore_index:
                                continue
                            try:
                                pi = int(p)
                            except Exception:
                                continue
                            flat_preds.append(pi)
                            flat_labels.append(li)

                    if flat_labels:
                        correct = sum(int(p == l) for p, l in zip(flat_preds, flat_labels))
                        token_acc = correct / max(len(flat_labels), 1)
                        teacher_metrics = {
                            'accuracy': float(token_acc),
                            'f1': float(token_acc),  # placeholder for LM mode summary compatibility
                            'precision': float(token_acc),
                            'recall': float(token_acc),
                        }
                    else:
                        teacher_metrics = {
                            'accuracy': 0.0,
                            'f1': 0.0,
                            'precision': 0.0,
                            'recall': 0.0,
                        }
                else:
                    teacher_metrics = compute_all_metrics(all_preds, all_labels)
                # Save last preds/labels for confusion matrix later
                self.teacher_last_preds = all_preds
                self.teacher_last_labels = all_labels
                # Track for comparison plot
                for mk in self.teacher_metrics_history.keys():
                    self.teacher_metrics_history[mk].append(teacher_metrics.get(mk, 0.0))
            
            accuracy = teacher_metrics.get('accuracy', 0)
            f1 = teacher_metrics.get('f1', 0)
            
            print(f"[TEACHER] Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, "
                  f"Val Loss={avg_val_loss:.4f}, Accuracy={accuracy:.4f}, F1={f1:.4f}")
            
            # Save best teacher
            if avg_val_loss < best_teacher_loss:
                best_teacher_loss = avg_val_loss
                best_teacher_state = copy.deepcopy(self.teacher.state_dict())
                print(f"[TEACHER] Best model updated (val_loss={avg_val_loss:.4f})")
            
            self.teacher.train()
        
        # Restore best teacher
        if best_teacher_state:
            self.teacher.load_state_dict(best_teacher_state)
            print(f"[TEACHER] Restored best teacher model (val_loss={best_teacher_loss:.4f})")
        # After teacher training, plot aggregate training curves
        try:
            teacher_curve_path = os.path.join(self.experiment_dir, 'teacher_training_curves.png')
            plot_training_curves(
                self.teacher_epoch_losses,
                self.teacher_epoch_val_losses,
                {},
                teacher_curve_path,
                lr_history=None,
                batch_train_losses=self.teacher_batch_train_losses,
                batch_val_losses=self.teacher_batch_val_losses,
                batch_val_running_acc=self.teacher_batch_val_running_acc,
            )
            print(f"[TEACHER] Training curves saved to {teacher_curve_path}")
        except Exception as e:
            print(f"[WARNING] Failed to plot teacher training curves: {e}")

        # Set teacher back to eval mode for distillation
        self.teacher.eval()
        # Flush CSV
        if self._csv_writer and hasattr(self, '_csv_file_handle'):
            try:
                self._csv_file_handle.flush()
            except Exception:
                pass

    def _maybe_trigger_overfit_guard(self, epoch_idx: int, train_loader=None) -> bool:
        """React to emerging overfitting either by mitigation or halt."""
        cfg = self.overfit_guard_config
        if not cfg['enabled']:
            return False

        min_epochs = cfg['min_epochs']
        if len(self.train_losses) < min_epochs or len(self.val_losses) < min_epochs:
            return False

        analysis = OverfitUnderfitDetector.analyze_training_curves(
            self.train_losses,
            self.val_losses,
            None,
            self.metrics_history.get('accuracy', []),
            metric_name='accuracy'
        )

        status = analysis.get('status', 'healthy')
        confidence = float(analysis.get('confidence', 0.0))
        loss_gap = analysis.get('loss_gap_pct', 0.0)
        status_label = status.replace('_', ' ').title()
        should_react = status in ('overfitting', 'mild_overfitting') and confidence >= cfg['confidence_threshold']

        if status in ('overfitting', 'mild_overfitting'):
            print(f"[OVERFIT-GUARD] {status_label} detected at epoch {epoch_idx} (confidence {confidence:.2f}).")

        event_record = {
            'epoch': epoch_idx,
            'status': status,
            'confidence': confidence,
            'loss_gap_pct': loss_gap,
            'analysis': analysis
        }

        if should_react:
            mitigation_actions = self._apply_overfit_mitigation(epoch_idx, analysis, train_loader)
            if mitigation_actions:
                event_record['action'] = 'mitigation'
                event_record['interventions'] = mitigation_actions
                self.overfit_guard_state['events'].append(event_record)
                self.overfit_guard_state.update({
                    'triggered': False,
                    'epoch': epoch_idx,
                    'status': status,
                    'confidence': confidence,
                    'loss_gap_pct': loss_gap,
                    'analysis': analysis,
                    'action': 'mitigation'
                })
                print(f"[OVERFIT-MITIGATION] Applied interventions: {', '.join(mitigation_actions)}.")
                print("[OVERFIT-MITIGATION] Continuing training to measure impact before considering a halt.")
                return False

            self.overfit_guard_state.update({
                'triggered': True,
                'epoch': epoch_idx,
                'status': status,
                'confidence': confidence,
                'loss_gap_pct': loss_gap,
                'analysis': analysis,
                'action': 'halt'
            })
            event_record['action'] = 'halt'
            self.overfit_guard_state['events'].append(event_record)
            if cfg['mode'] in ('early_stop', 'stop', 'halt'):
                print("[OVERFIT-GUARD] Halting student training to prevent further overfitting.")
                return True
            print("[OVERFIT-GUARD] Guard mode set to 'warn'; training will continue.")
            return False

        if status in ('overfitting', 'mild_overfitting'):
            event_record['action'] = 'observe'
            self.overfit_guard_state['events'].append(event_record)
            self.overfit_guard_state.update({
                'triggered': False,
                'epoch': epoch_idx,
                'status': status,
                'confidence': confidence,
                'loss_gap_pct': loss_gap,
                'analysis': analysis,
                'action': 'observe'
            })

        return False

    def resume_from_checkpoint(self, checkpoint_dir: str):
        """Resume training from a saved checkpoint."""
        from pathlib import Path
        from zynthe.core.models.model_saver import load_checkpoint
        from zynthe.core.models.model_saver import CheckpointMetadata
        
        ckpt_path = Path(checkpoint_dir)
        if not ckpt_path.exists():
            print(f"[WARNING] Checkpoint directory {checkpoint_dir} does not exist. Starting from scratch.")
            return
            
        print(f"[INFO] Resuming training from checkpoint: {checkpoint_dir}")
        
        # Load model/tokenizer artifacts if present.
        try:
            model_file = ckpt_path / "pytorch_model.bin"
            if not model_file.exists():
                model_file = ckpt_path / "model.safetensors"

            if model_file.exists():
                print(f"[INFO] Loading model weights from {model_file}")
                self.student = self.student.__class__.from_pretrained(str(ckpt_path)).to(self.device)
            else:
                print(f"[WARNING] No model weights found in {checkpoint_dir}")
        except Exception as e:
            print(f"[WARNING] Failed to load model weights from checkpoint: {e}")

        # Restore full optimizer/scheduler/scaler state when checkpoint payload exists.
        state_candidates = [
            ckpt_path / "checkpoint.pt",
            ckpt_path / "latest.pt",
            ckpt_path / "best.pt",
        ]
        state_path = next((candidate for candidate in state_candidates if candidate.exists()), None)
        if state_path is not None:
            try:
                payload, metadata = load_checkpoint(
                    model=self.student,
                    optimizer=self.optimizer,
                    path=str(state_path),
                    scheduler=self.scheduler,
                    scaler=self.scaler,
                    map_location=str(self.device),
                    strict=False,
                )
                print(f"[INFO] Restored optimizer/scheduler state from {state_path}")
                if isinstance(metadata, CheckpointMetadata):
                    self.best_val_loss = float(metadata.best_metric) if metadata.best_metric is not None else self.best_val_loss
                    self.resume_epoch = int(max(0, metadata.epoch))
                    self.resume_global_step = int(max(0, metadata.global_step))
                    if metadata.epoch > 0:
                        print(f"[INFO] Resumed metadata: epoch={metadata.epoch}, step={metadata.global_step}")
                else:
                    # Backward compatibility with legacy top-level checkpoint fields.
                    if isinstance(payload, dict):
                        self.resume_epoch = int(max(0, int(payload.get('epoch', 0) or 0)))
                        self.resume_global_step = int(max(0, int(payload.get('global_step', 0) or 0)))
                        best_metric = payload.get('best_metric')
                        if isinstance(best_metric, (int, float)):
                            self.best_val_loss = float(best_metric)
            except Exception as e:
                print(f"[WARNING] Failed to restore checkpoint training state: {e}")
