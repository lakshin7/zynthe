import copy
import csv
from datetime import datetime
from evaluation.metrics import compute_all_metrics
from evaluation.visualizer import (
    plot_training_curves,
    plot_teacher_student_comparison,
    plot_epoch_micro_series,
    plot_metric_grid,
    plot_calibration_curve,
    plot_runtime_profile,
)
from evaluation.metrics import plot_metrics
from evaluation.metrics_extended import (
    compute_extended_metrics,
    DistillationEfficacyIndex,
    CompressionAwareScore,
    LossComponentTracker
)
from core.distillers.multi_stage_distiller import DistillerRegistry
from training.optimizer import OptimizerFactory, GradientManager, AdaptiveOptimizer
from training.scheduler import SchedulerFactory
from core.utils.data_validator import DataValidator, DataLeakageDetector, OverfitUnderfitDetector
import torch
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler  # Mixed Precision Training
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
    def __init__(self, teacher, student, tokenizer, config, device, experiment_dir, websocket_callback: Optional[Callable[[Dict[str, Any]], None]] = None):
        self.teacher = teacher
        self.student = student
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        self.experiment_dir = experiment_dir
        
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
            LOG.warning("⚠️  TEACHER MODEL WARNING: Base model detected!")
            LOG.warning(f"   Teacher: {teacher_model_name}")
            LOG.warning("   Base models are NOT task-trained and will perform poorly!")
            LOG.warning("   This leads to low-quality distillation (teacher < student).")
            LOG.warning("")
            LOG.warning("   RECOMMENDATIONS:")
            LOG.warning("   1. Use a task-specific fine-tuned teacher model, OR")
            LOG.warning("   2. Enable teacher training: set train_teacher: true in config")
            LOG.warning("=" * 80)
            print(f"\n⚠️  WARNING: Using base model '{teacher_model_name}' as teacher.")
            print("   This may result in poor distillation. Consider using a fine-tuned model.")
            print("   See logs for recommendations.\n")
        
        # ========== END TEACHER MODEL VALIDATION ==========
        
        # ========== PERFORMANCE OPTIMIZATIONS ==========
        
        # 1. Mixed Precision Training (AMP) - 2-3x speedup
        self.use_amp = self.config['train'].get('use_amp', True)
        self.scaler = GradScaler() if self.use_amp else None
        if self.use_amp:
            print("[OPTIMIZATION] Mixed Precision Training (AMP) enabled - expect 2-3x speedup")
        
        # 2. Gradient Accumulation - simulate larger batch sizes
        self.gradient_accumulation_steps = self.config['train'].get('gradient_accumulation_steps', 1)
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
        num_epochs = self.config['train'].get('num_epochs', 10)
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
            self.teacher_optimizer = AdamW(self.teacher.parameters(), lr=self.config['train'].get('teacher_lr', 2e-5))
        
        # Initialize distiller from registry
        distil_cfg = self.config['distillation']
        registry = DistillerRegistry()
        distiller_type = distil_cfg.get('type', 'kd')
        
        # Get distiller class and instantiate with proper parameters
        distiller_class = registry.get(distiller_type)
        
        # Check if distiller_class is valid before instantiating
        if distiller_class is None:
            raise ValueError(f"Unknown distiller type: {distiller_type}")
        
        distiller_config = distil_cfg.get('config', {})
        
        self.distiller = distiller_class(
            teacher=self.teacher,
            student=self.student,
            device=self.device,
            **distiller_config
        )
        
        self.train_losses = []
        self.val_losses = []
        self.metrics_history = {'accuracy': [], 'f1': [], 'precision': [], 'recall': []}
        # Optional batch-level tracking for detailed visualization
        self.batch_train_losses = []  # List[List[float]] per epoch
        self.batch_val_losses = []    # List[List[float]] per epoch
        self.batch_val_running_acc = []  # List[List[float]] per epoch
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.early_stop_patience = self.config['train'].get('early_stop_patience', 2)
        self.no_improve_epochs = 0
        self.last_preds = []
        self.last_labels = []

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
        self.extended_metrics_history = {
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
        self.teacher_metrics_history = {'accuracy': [], 'f1': [], 'precision': [], 'recall': []}

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
            
            # ========== MIXED PRECISION TRAINING (AMP) ==========
            # Wrap forward pass in autocast for automatic mixed precision
            # Note: MPS doesn't support autocast, use CPU dtype
            amp_dtype = torch.bfloat16 if self.use_amp and self.device.type == 'cpu' else torch.float16
            with torch.no_grad():
                if self.use_amp and self.device.type != 'mps':
                    with autocast(dtype=amp_dtype):
                        try:
                            teacher_outputs = self.teacher(**teacher_batch)
                        except Exception as e:
                            print(f"[WARNING] Teacher forward pass failed at batch {batch_idx}: {e}")
                            continue
                else:
                    try:
                        teacher_outputs = self.teacher(**teacher_batch)
                    except Exception as e:
                        print(f"[WARNING] Teacher forward pass failed at batch {batch_idx}: {e}")
                        continue
            
            if self.use_amp and self.device.type != 'mps':
                with autocast(dtype=amp_dtype):
                    try:
                        student_outputs = self.student(**student_batch)
                    except Exception as e:
                        print(f"[WARNING] Student forward pass failed at batch {batch_idx}: {e}")
                        continue
                        
                    labels = batch.get('labels', None)
                    try:
                        result = self.distiller.compute_loss(
                            student_outputs=student_outputs,
                            teacher_outputs=teacher_outputs,
                            targets=labels
                        )
                        
                        # Handle tuple return (loss, metrics_dict)
                        if isinstance(result, tuple):
                            loss, metrics_dict = result
                        else:
                            # Fallback if distiller returns only loss
                            loss = result
                            metrics_dict = {}
                        
                        # Scale loss for gradient accumulation
                        loss = loss / self.gradient_accumulation_steps
                            
                    except Exception as e:
                        print(f"[WARNING] Loss computation failed at batch {batch_idx}: {e}")
                        loss = torch.tensor(0.0, device=self.device)
            else:
                try:
                    student_outputs = self.student(**student_batch)
                except Exception as e:
                    print(f"[WARNING] Student forward pass failed at batch {batch_idx}: {e}")
                    continue
                    
                labels = batch.get('labels', None)
                try:
                    result = self.distiller.compute_loss(
                        student_outputs=student_outputs,
                        teacher_outputs=teacher_outputs,
                        targets=labels
                    )
                    
                    # Handle tuple return (loss, metrics_dict)
                    if isinstance(result, tuple):
                        loss, metrics_dict = result
                    else:
                        # Fallback if distiller returns only loss
                        loss = result
                        metrics_dict = {}
                    
                    # Scale loss for gradient accumulation
                    loss = loss / self.gradient_accumulation_steps
                        
                except Exception as e:
                    print(f"[WARNING] Loss computation failed at batch {batch_idx}: {e}")
                    loss = torch.tensor(0.0, device=self.device)
            
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
            
            # Step scheduler per-batch for all schedulers EXCEPT ReduceLROnPlateau (epoch-based)
            if self.scheduler is not None and hasattr(self.scheduler, 'step'):
                scheduler_name = type(self.scheduler).__name__
                if 'ReduceLROnPlateau' not in scheduler_name:
                    # Type safety: Check if step method requires metrics parameter
                    step_signature = inspect.signature(self.scheduler.step)
                    if 'metrics' in step_signature.parameters:
                        self.scheduler.step(metrics=None)  # type: ignore[call-arg]
                    else:
                        self.scheduler.step()  # type: ignore[call-arg]
            
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

    def evaluate(self, dataloader, compute_extended=True):
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
                
                try:
                    teacher_outputs = self.teacher(**teacher_batch)
                except Exception as e:
                    print(f"[WARNING] Teacher forward pass failed at batch {batch_idx} during evaluation: {e}")
                    failed_batches += 1
                    continue
                forward_start = time.perf_counter()
                try:
                    student_outputs = self.student(**student_batch)
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
                try:
                    result = self.distiller.compute_loss(
                        student_outputs=student_outputs,
                        teacher_outputs=teacher_outputs,
                        targets=labels
                    )
                    
                    # Handle tuple return (loss, metrics_dict)
                    if isinstance(result, tuple):
                        loss, metrics_dict = result
                    else:
                        # Fallback if distiller returns only loss
                        loss = result
                        metrics_dict = {}
                        
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
                        if student_logits.dim() >= 2:
                            preds = torch.argmax(student_logits, dim=-1)
                        else:
                            preds = (student_logits > 0).long()
                        preds_list = preds.cpu().numpy().tolist()
                        labels_list = labels.cpu().numpy().tolist()
                        all_preds.extend(preds_list)
                        all_labels.extend(labels_list)

                        prob_tensor = None
                        if student_logits.dim() >= 2:
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

                        # Update running accuracy
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

        if all_labels and all_preds and len(all_labels) == len(all_preds):
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
                extended_metrics = compute_extended_metrics(
                    teacher_logits_cat, 
                    student_logits_cat,
                    temperature=temperature
                )
                
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

        return avg_loss, metrics, extended_metrics, details

    def _build_eval_diagnostics(
        self,
        metrics: Dict[str, Any],
        avg_loss: Optional[float],
        batch_losses: List[float],
        confidence_samples: List[float],
        probabilities: Optional[np.ndarray],
    ) -> Dict[str, Any]:
        diagnostics: Dict[str, Any] = {}
        warnings: List[str] = []

        if avg_loss is not None:
            diagnostics['eval_loss'] = float(avg_loss)

        core_keys = ['accuracy', 'precision', 'recall', 'f1']
        core_vals = [metrics.get(key) for key in core_keys]
        numeric_vals = [float(val) for val in core_vals if isinstance(val, (int, float, np.floating))]
        if numeric_vals:
            spread = max(numeric_vals) - min(numeric_vals)
            diagnostics['core_metric_spread'] = float(spread)
            if spread < 0.01:
                warnings.append('metrics_overlap')

        if batch_losses:
            losses_arr = np.asarray(batch_losses, dtype=np.float64)
            diagnostics['loss_std'] = float(losses_arr.std())
            diagnostics['loss_min'] = float(losses_arr.min())
            diagnostics['loss_max'] = float(losses_arr.max())

        if confidence_samples:
            conf_arr = np.asarray(confidence_samples, dtype=np.float64)
            diagnostics['confidence_mean'] = float(conf_arr.mean())
            diagnostics['confidence_std'] = float(conf_arr.std())
            diagnostics['confidence_min'] = float(conf_arr.min())
            diagnostics['confidence_max'] = float(conf_arr.max())
            if conf_arr.mean() > 0.95 and conf_arr.std() < 0.02:
                warnings.append('confidence_collapse')

        if probabilities is not None and probabilities.ndim >= 1:
            try:
                probs = probabilities
                if probabilities.ndim == 1:
                    probs = probabilities.reshape(-1, 1)
                entropy = -np.sum(probs * np.log(np.clip(probs, 1e-9, 1.0)), axis=1)
                diagnostics['prediction_entropy_mean'] = float(np.mean(entropy))
                diagnostics['prediction_entropy_std'] = float(np.std(entropy))
                if np.mean(entropy) < 0.2:
                    warnings.append('low_entropy_predictions')
            except Exception:
                LOG.debug("Failed to compute entropy for diagnostics", exc_info=True)

        diagnostics['warnings'] = warnings
        return diagnostics

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
                        LOG.error(f"  ❌ {error}")
                    LOG.error("\nPlease fix data issues before training.")
                    LOG.error("See report: " + str(validation_path))
                    LOG.error("="*80)
                    
                    # Ask user to confirm
                    user_input = input("\nData validation failed. Continue anyway? (yes/no): ")
                    if user_input.lower() != 'yes':
                        raise ValueError("Training aborted due to data validation failure")
                    else:
                        LOG.warning("User chose to continue despite validation failure")
        
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
            LOG.info(f"Initializing scheduler with:")
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
        
        # Optional: Train teacher first if configured
        if self.should_train_teacher:
            print(f"\n{'='*70}")
            print(f"PHASE 2.5: Fine-tuning Teacher Model")
            print(f"{'='*70}\n")
            print(f"[INFO] Training teacher for {self.teacher_epochs} epochs before distillation...")
            self._train_teacher(train_loader, val_loader)
            print(f"[INFO] Teacher training completed. Starting distillation...\n")
        
        metrics_history = []
        epochs = self.config['train']['epochs']
        print(f"[INFO] Training started for {epochs} epochs.")
        for epoch in range(epochs):
            print(f"[INFO] Starting epoch {epoch+1}/{epochs}")
            train_loss = self.train_epoch(train_loader)
            val_loss, val_metrics, extended, val_details = self.evaluate(val_loader, compute_extended=True)
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
                    break

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
            self.student.save_pretrained(student_save_dir)
            self.tokenizer.save_pretrained(student_save_dir)
            print(f'[INFO] Student model and tokenizer saved to {student_save_dir}')
            
            # Save teacher model for comparison
            teacher_save_dir = os.path.join(self.experiment_dir, 'teacher_model')
            self.teacher.save_pretrained(teacher_save_dir)
            self.tokenizer.save_pretrained(teacher_save_dir)
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
            from evaluation.metrics import plot_metrics, compute_all_metrics  # type: ignore
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
                from evaluation.metrics import compute_all_metrics as _cm_all  # type: ignore
                from evaluation.metrics import plot_metrics as _plot_metrics  # type: ignore
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

        # Teacher vs Student comparison plot (if enabled and data exists)
        if self.enable_comparison_plot and self.teacher_epoch_losses and self.train_losses:
            try:
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
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs['loss']
                
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
                    loss = outputs.loss if hasattr(outputs, 'loss') else outputs['loss']
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
