"""
Preflight Analyzer - Main Orchestrator
=======================================

Coordinates all preflight checks before distillation:
1. Model inspection (compatibility, architecture, parameters)
2. Data validation (schema, task type, distribution)
3. Resource probing (devices, memory, precision)
4. Auto-configuration (batch size, precision, device)
5. Comprehensive reporting (go/no-go decision)

Usage:
    analyzer = PreflightAnalyzer(
        teacher_model=teacher,
        student_model=student,
        dataset=train_dataset,
        config=config
    )
    
    report = analyzer.run_preflight()
    
    if report['can_proceed']:
        # Start training with optimized config
        optimized_config = report['optimized_config']
    else:
        # Fix issues
        logger.info(report['blockers'])
"""

from __future__ import annotations


import math
from typing import Dict, List, Optional, Any
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import yaml
import json
from pathlib import Path
from datetime import datetime

from .model_inspector import ModelInspector
from .data_inspector import DataInspector
from .resource_probe import ResourceProbe

try:  # Optional dependency on distiller presets
    from zynthe.core.distillers.presets import get_preset, list_presets  # type: ignore
except Exception:  # pragma: no cover - presets may be unavailable
    get_preset = None  # type: ignore
    list_presets = None  # type: ignore

import logging

logger = logging.getLogger(__name__)


class PreflightAnalyzer:
    """
    Main orchestrator for all preflight checks.
    
    Runs comprehensive analysis before distillation:
    - Model compatibility checking
    - Data validation
    - Resource profiling
    - Configuration optimization
    - Go/no-go decision with detailed reasoning
    """
    
    def __init__(
        self,
        teacher_model: Optional[nn.Module] = None,
        student_model: Optional[nn.Module] = None,
        dataset: Optional[Dataset] = None,
        config: Optional[Dict] = None,
        output_dir: Optional[str] = None
    ):
        """
        Initialize preflight analyzer.
        
        Args:
            teacher_model: Teacher model
            student_model: Student model
            dataset: Training dataset
            config: Configuration dictionary
            output_dir: Directory to save reports
        """
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.dataset = dataset
        self.config = config or {}
        self.output_dir = Path(output_dir) if output_dir else Path("preflight_reports")
        
        # Initialize inspectors
        self.model_inspector = ModelInspector(teacher_model, student_model)
        self.data_inspector = DataInspector(dataset, config)
        self.resource_probe = ResourceProbe()
        
        # Results storage
        self.results: Dict[str, Any] = {}
    
    def validate_config(self) -> Dict[str, Any]:
        """
        Validate configuration structure before model loading.
        
        This is Step 1.1 in the workflow - catches config errors early.
        
        Returns:
            Validation report with errors and warnings
        """
        validation: Dict[str, Any] = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'info': []
        }
        
        if not self.config:
            validation['is_valid'] = False
            validation['errors'].append("No configuration provided")
            return validation
        
        # Check model configuration
        model_cfg = self.config.get('model', {})
        
        if not model_cfg.get('name'):
            validation['is_valid'] = False
            validation['errors'].append("Missing 'model.name' (teacher model) in config")
        else:
            validation['info'].append(f"Teacher model: {model_cfg['name']}")
        
        if not model_cfg.get('student_name'):
            validation['warnings'].append(
                "Missing 'model.student_name' - will default to teacher model (no compression)"
            )
        else:
            validation['info'].append(f"Student model: {model_cfg['student_name']}")
        
        if not model_cfg.get('type'):
            validation['warnings'].append("Missing 'model.type' - will default to AutoModel")
        
        # Check data configuration
        data_cfg = self.config.get('data', {})
        
        if not data_cfg.get('train_path'):
            validation['is_valid'] = False
            validation['errors'].append("Missing 'data.train_path' in config")
        else:
            train_path = Path(data_cfg['train_path'])
            if not train_path.exists():
                validation['is_valid'] = False
                validation['errors'].append(f"Training data not found: {train_path}")
            else:
                validation['info'].append(f"Training data: {train_path} ")
        
        if not data_cfg.get('val_path'):
            validation['warnings'].append("Missing 'data.val_path' - validation will be skipped")
        else:
            val_path = Path(data_cfg['val_path'])
            if not val_path.exists():
                validation['warnings'].append(f"Validation data not found: {val_path}")
            else:
                validation['info'].append(f"Validation data: {val_path} ")
        
        # Check distillation configuration
        distill_cfg = self.config.get('distillation', {})
        valid_methods = ['kd_hinton', 'feature', 'attention', 'similarity', 'multi_stage']
        
        if distill_cfg.get('method') and distill_cfg['method'] not in valid_methods:
            validation['warnings'].append(
                f"Unknown distillation method '{distill_cfg['method']}'. "
                f"Valid: {', '.join(valid_methods)}"
            )
        
        # Check device configuration
        device_cfg = self.config.get('device', {})
        
        if device_cfg.get('prefer_cuda') and not torch.cuda.is_available():
            validation['warnings'].append("Config prefers CUDA but CUDA not available")
        
        if device_cfg.get('prefer_mps') and not torch.backends.mps.is_available():
            validation['warnings'].append("Config prefers MPS but MPS not available")
        
        # Check training configuration
        train_cfg = self.config.get('train', {})
        
        if train_cfg.get('batch_size', 0) > 128:
            validation['warnings'].append(
                f"Very large batch size ({train_cfg['batch_size']}) may cause OOM"
            )
        
        if train_cfg.get('batch_size', 0) < 1:
            validation['is_valid'] = False
            validation['errors'].append("Invalid batch size (must be >= 1)")

        # Overfit guard visibility so users know the new safeguard is active
        guard_cfg = train_cfg.get('overfit_guard', True)
        if isinstance(guard_cfg, dict):
            guard_enabled = guard_cfg.get('enabled', True)
            guard_mode = guard_cfg.get('mode', guard_cfg.get('action', 'early_stop'))
        else:
            guard_enabled = bool(guard_cfg)
            guard_mode = 'early_stop'

        if guard_enabled:
            validation['info'].append(f"Overfit guard enabled (mode: {guard_mode})")
        else:
            validation['warnings'].append(
                "train.overfit_guard is disabled - training will not auto-halt when validation loss diverges"
            )

        mitigation_cfg = train_cfg.get('overfit_mitigation', True)
        if isinstance(mitigation_cfg, dict):
            mitigation_enabled = mitigation_cfg.get('enabled', True)
        elif mitigation_cfg in (False, None):
            mitigation_enabled = False
        else:
            mitigation_enabled = True

        if mitigation_enabled:
            validation['info'].append("Adaptive overfit mitigation enabled - staged regularization will run before hard stops")
        else:
            validation['warnings'].append(
                "train.overfit_mitigation is disabled - guard will only stop runs without attempting regularization"
            )

        if not train_cfg.get('train_teacher', False):
            validation['warnings'].append(
                "Teacher fine-tuning (train.train_teacher) is disabled. Distillation quality may drop without a tuned teacher."
            )
        else:
            validation['info'].append("Teacher fine-tuning enabled before distillation")
        
        return validation
    
    def run_preflight(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Run complete preflight analysis.
        
        Args:
            verbose: Print reports during execution
            
        Returns:
            Comprehensive analysis report with go/no-go decision
        """
        logger.info("=" * 70)
        logger.info("RUNNING PREFLIGHT ANALYSIS")
        logger.info("=" * 70)
        print()
        
        # 0. Config Validation (Phase 1.1 - NEW)
        logger.info("[CHECK] Validating configuration...")
        config_validation = self.validate_config()
        self.results['config_validation'] = config_validation
        
        if verbose:
            logger.info(self._format_config_validation(config_validation))
            print()
        
        if not config_validation['is_valid']:
            logger.error("[FAIL] Config validation failed. Cannot proceed.")
            return {
                'timestamp': datetime.now().isoformat(),
                'can_proceed': False,
                'confidence': 'none',
                'blockers': config_validation['errors'],
                'warnings': config_validation['warnings'],
                'recommendations': ['Fix config errors before proceeding'],
                'config_validation': config_validation
            }
        
        # 1. Model Inspection
        logger.info("[LIST] Inspecting models...")
        model_report = self.model_inspector.inspect()
        self.results['model'] = model_report
        
        if verbose:
            logger.info(self.model_inspector.generate_report())
            print()
        
        # 2. Data Inspection
        logger.info("[INFO] Inspecting dataset...")
        data_report = self.data_inspector.validate()
        
        # Modality check: Enforce text-only for core stabilization
        if data_report.get('modality') not in ('text', 'nlp', 'unknown') and data_report.get('status') == 'pass':
            data_report['status'] = 'fail'
            data_report['issues'].append({
                'severity': 'critical',
                'component': 'data',
                'message': f"Zynthé currently strictly supports text/nlp modalities. Found: {data_report.get('modality')}"
            })
            
        self.results['data'] = data_report
        
        if verbose:
            logger.info(self.data_inspector.generate_report())
            print()
        
        # 3. Resource Probing
        logger.info("[CHECK] Probing hardware resources...")
        resource_report = self.resource_probe.probe()
        self.results['resources'] = resource_report
        
        if verbose:
            logger.info(self.resource_probe.generate_report())
            print()
        
        # 4. Cross-validate and optimize
        logger.info("  Optimizing configuration...")
        optimization_report = self._optimize_configuration()
        self.results['optimization'] = optimization_report
        
        # 5. Make go/no-go decision
        logger.info("[TARGET] Evaluating readiness...")
        decision = self._make_decision()
        self.results['decision'] = decision

        # 6. Assess overall risk and create an action plan
        risk_profile = self._compute_risk_profile()
        self.results['risk_profile'] = risk_profile
        playbook = self._generate_user_playbook()
        self.results['playbook'] = playbook
        
        # 7. Generate comprehensive report
        comprehensive_report = self._generate_comprehensive_report()
        
        if verbose:
            logger.info(self._format_final_report(comprehensive_report))
        return comprehensive_report
    
    def _optimize_configuration(self) -> Dict[str, Any]:
        """
        Optimize configuration based on all inspections.
        
        Returns:
            Optimized configuration
        """
        optimization: Dict[str, Any] = {
            'device': None,
            'precision': None,
            'batch_size': None,
            'num_workers': None,
            'pin_memory': None,
            'distillation_strategy': None,
            'layer_mapping': None,
            'use_amp': False,
            'changes': []
        }
        
        # Get recommendations from each inspector
        resource_rec = self.results['resources']['recommendations']
        model_rec = self.results['model'].get('recommended_strategy', {})
        data_rec = self.results['data'].get('batch_recommendations', {})
        
        # Device selection
        optimization['device'] = resource_rec['device']
        optimization['changes'].append(f"Device: {optimization['device']}")
        
        # Precision selection
        optimization['precision'] = resource_rec['precision']
        optimization['use_amp'] = resource_rec['use_amp']
        optimization['changes'].append(
            f"Precision: {optimization['precision']} (AMP: {optimization['use_amp']})"
        )
        
        # Batch size optimization
        base_batch = data_rec.get('optimal_batch_size', 32)
        multiplier = resource_rec.get('batch_size_multiplier', 1.0)
        optimal_batch = int(base_batch * multiplier)

        # Ensure batch size is power of 2 for efficiency when possible
        if optimal_batch > 0:
            optimal_batch = 2 ** int(math.floor(math.log2(optimal_batch))) if optimal_batch > 1 else 1
        optimization['batch_size'] = max(1, optimal_batch)
        optimization['changes'].append(
            f"Batch size: {optimization['batch_size']} "
            f"(base: {base_batch}, multiplier: {multiplier}x)"
        )
        
        # DataLoader workers
        optimization['num_workers'] = resource_rec['num_workers']
        optimization['pin_memory'] = resource_rec['pin_memory']
        optimization['changes'].append(
            f"Workers: {optimization['num_workers']}, Pin memory: {optimization['pin_memory']}"
        )
        
        # Distillation strategy
        if model_rec:
            optimization['distillation_strategy'] = model_rec
            optimization['changes'].append(
                f"Strategy: {model_rec.get('primary_method', 'unknown')}"
            )
        
        # Layer mapping
        layer_mapping_info = self.results['model'].get('layer_mapping', {})
        if isinstance(layer_mapping_info, dict) and layer_mapping_info.get('mappings'):
            optimization['layer_mapping'] = layer_mapping_info['mappings']
            optimization['changes'].append(
                f"Layer mapping: {len(layer_mapping_info['mappings'])} pairs auto-mapped"
            )

        preset_info = self._derive_recommended_preset()
        if preset_info:
            optimization['recommended_preset'] = preset_info
            preset_name = preset_info['name']
            optimization['changes'].append(f"Preset selected: {preset_name}")
        
        # Memory estimation
        if self.student_model:
            student_params = self.results['model'].get('student', {}).get('total_params', 0)
            available_memory = self._get_available_memory(optimization['device'])
            
            if available_memory:
                memory_rec = self.resource_probe.recommend_optimal_batch_size(
                    student_params,
                    available_memory,
                    sequence_length=self._estimate_sequence_length(),
                    precision=optimization['precision']
                )
                
                optimization['memory_estimate'] = memory_rec
                
                # Adjust batch size if needed
                if memory_rec['optimal_batch_size'] < optimization['batch_size']:
                    optimization['batch_size'] = memory_rec['optimal_batch_size']
                    optimization['changes'].append(
                        f"Batch size reduced to {optimization['batch_size']} "
                        f"due to memory constraints"
                    )
        
        return optimization

    def _derive_recommended_preset(self) -> Optional[Dict[str, Any]]:
        """Select a distillation preset based on analysis heuristics."""
        if get_preset is None:
            return None

        compression = self.results.get('model', {}).get('compression_ratio', 1.0)
        data_type = self.results.get('data', {}).get('data_type', 'unknown')
        model_arch = self.results.get('model', {}).get('teacher', {}).get('architecture_family', 'unknown')

        preset_choice = 'quick_start'
        rationale = []

        if compression and compression > 8:
            preset_choice = 'compression_max'
            rationale.append('High compression ratio detected (>8x).')
        elif model_arch == 'transformer' and data_type == 'text':
            preset_choice = 'balanced'
            rationale.append('Transformer models on text tasks benefit from balanced preset.')
        elif data_type in {'vision', 'video'}:
            preset_choice = 'vision_transformer'
            rationale.append('Vision dataset detected; using vision-specific preset.')
        elif compression and compression < 2:
            preset_choice = 'quick_start'
            rationale.append('Low compression ratio; quick start preset is sufficient.')

        try:
            preset_config = get_preset(preset_choice)
            return {
                'name': preset_choice,
                'config': preset_config,
                'rationale': rationale or [f"Heuristic selection for data type {data_type}."]
            }
        except KeyError:
            if list_presets:  # type: ignore[truthy-function]
                available = list_presets()
                return {
                    'name': preset_choice,
                    'config': {},
                    'rationale': [
                        f"Preset '{preset_choice}' not found. Available presets: {available}"
                    ]
                }
            return None
    
    def _get_available_memory(self, device: str) -> Optional[float]:
        """Get available memory for device in GB."""
        memory_info = self.results['resources']['memory']
        
        if device == 'cuda':
            if memory_info['gpu']:
                return memory_info['gpu'][0].get('free', memory_info['gpu'][0]['total'])
        
        elif device == 'mps':
            # MPS uses system memory
            return memory_info['system']['available']
        
        elif device == 'cpu':
            return memory_info['system']['available']
        
        return None
    
    def _estimate_sequence_length(self) -> Optional[int]:
        """Estimate sequence length from data."""
        data_info = self.results['data'].get('dataset_info', {})
        sample_structure = data_info.get('sample_structure', {})
        
        # Look for sequence dimensions
        for field, props in sample_structure.get('fields', {}).items():
            shape = props.get('shape', [])
            if len(shape) >= 2 and shape[0] > 10:  # Likely sequence
                return shape[0]
        
        # Default for transformers
        data_type = self.results['data'].get('data_type')
        if data_type == 'text':
            return 512  # Common max length
        
        return None
    
    def _make_decision(self) -> Dict[str, Any]:
        """
        Make go/no-go decision based on all checks.
        
        Returns:
            Decision with reasoning
        """
        decision: Dict[str, Any] = {
            'can_proceed': True,
            'blockers': [],
            'warnings': [],
            'recommendations': [],
            'confidence': 'high'
        }
        
        # Check for blockers
        model_report = self.results['model']
        data_report = self.results['data']
        train_cfg = self.config.get('train', {})
        guard_cfg = train_cfg.get('overfit_guard', True)
        if isinstance(guard_cfg, dict):
            guard_enabled = guard_cfg.get('enabled', True)
            guard_mode = guard_cfg.get('mode', guard_cfg.get('action', 'early_stop'))
        else:
            guard_enabled = bool(guard_cfg)
            guard_mode = 'early_stop'
        teacher_training_enabled = bool(train_cfg.get('train_teacher', False))
        mitigation_cfg = train_cfg.get('overfit_mitigation', True)
        if isinstance(mitigation_cfg, dict):
            mitigation_enabled = mitigation_cfg.get('enabled', True)
        elif mitigation_cfg in (False, None):
            mitigation_enabled = False
        else:
            mitigation_enabled = True
        
        # Model blockers
        model_compat = model_report.get('compatibility', {})
        if not model_compat.get('is_compatible', True):
            decision['can_proceed'] = False
            decision['blockers'].extend(model_compat.get('errors', []))
        
        # Data blockers
        if not data_report['is_valid']:
            decision['can_proceed'] = False
            decision['blockers'].extend(data_report.get('errors', []))
        
        # Collect warnings
        decision['warnings'].extend(model_compat.get('warnings', []))
        decision['warnings'].extend(data_report.get('warnings', []))

        if not guard_enabled:
            warning = (
                "Overfit guard disabled - enable train.overfit_guard.enabled to automatically pause when validation loss spikes"
            )
            if warning not in decision['warnings']:
                decision['warnings'].append(warning)
            decision['recommendations'].append(
                "Enable the overfit guard or add regularization (dropout, data augmentation) to control overfitting."
            )
        else:
            decision['recommendations'].append(
                f"Overfit guard active (mode={guard_mode}); review training_health.json after runs for guard events."
            )

        if mitigation_enabled:
            decision['recommendations'].append(
                "Adaptive overfit mitigation enabled; watch for OVERFIT-MITIGATION logs to confirm interventions."
            )
        else:
            mitigation_warning = (
                "Adaptive overfit mitigation disabled - the system will halt without attempting staged regularization"
            )
            decision['warnings'].append(mitigation_warning)
            decision['recommendations'].append(
                "Enable train.overfit_mitigation to let the trainer apply augmentation/dropout before halting."
            )

        if not teacher_training_enabled:
            teacher_warning = (
                "Teacher fine-tuning disabled - teacher weights will remain generic during distillation"
            )
            decision['warnings'].append(teacher_warning)
            decision['recommendations'].append(
                "Set train.train_teacher=true to warm up the teacher model before distillation."
            )
        else:
            decision['recommendations'].append(
                "Teacher fine-tuning enabled; keep checkpoints for reuse across experiments."
            )
        
        # Collect recommendations
        decision['recommendations'].extend(model_compat.get('recommendations', []))
        
        # Adjust confidence based on warnings
        if len(decision['warnings']) > 5:
            decision['confidence'] = 'medium'
        elif len(decision['warnings']) > 10:
            decision['confidence'] = 'low'
        
        # Add specific recommendations
        if decision['can_proceed']:
            decision['recommendations'].append(
                "All checks passed. Ready to start distillation."
            )
            
            # Add optimization recommendations
            opt = self.results['optimization']
            if opt.get('use_amp'):
                decision['recommendations'].append(
                    "Enable Automatic Mixed Precision (AMP) for faster training."
                )
            
            if opt.get('batch_size', 0) > 32:
                decision['recommendations'].append(
                    f"Large batch size ({opt['batch_size']}) detected. "
                    f"Consider using gradient accumulation if memory is limited."
                )
        
        return decision
    
    def _generate_comprehensive_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive report combining all results.
        
        Returns:
            Full report dictionary
        """
        return {
            'timestamp': datetime.now().isoformat(),
            'can_proceed': self.results['decision']['can_proceed'],
            'confidence': self.results['decision']['confidence'],
            'blockers': self.results['decision']['blockers'],
            'warnings': self.results['decision']['warnings'],
            'recommendations': self.results['decision']['recommendations'],
            'model_analysis': self.results['model'],
            'data_analysis': self.results['data'],
            'resource_profile': self.results['resources'],
            'optimized_config': self.results['optimization'],
            'risk_profile': self.results.get('risk_profile', {}),
            'playbook': self.results.get('playbook', {})
        }
    
    def _format_config_validation(self, validation: Dict[str, Any]) -> str:
        """Format config validation results."""
        lines = []
        
        if validation['is_valid']:
            lines.append("[OK] Configuration is valid")
        else:
            lines.append("[FAIL] Configuration has errors")
        
        if validation['errors']:
            lines.append("\n[!] ERRORS:")
            for error in validation['errors']:
                lines.append(f"  • {error}")
        
        if validation['warnings']:
            lines.append(f"\n  WARNINGS ({len(validation['warnings'])}):")
            for warning in validation['warnings'][:3]:
                lines.append(f"  • {warning}")
            if len(validation['warnings']) > 3:
                lines.append(f"  ... and {len(validation['warnings']) - 3} more")
        
        if validation['info']:
            lines.append("\nℹ  INFO:")
            for info in validation['info']:
                lines.append(f"  • {info}")
        
        return "\n".join(lines)
    
    def _format_final_report(self, report: Dict[str, Any]) -> str:
        """
        Format comprehensive report as human-readable string.
        
        Args:
            report: Report dictionary
            
        Returns:
            Formatted report string
        """
        lines = [
            "",
            "=" * 70,
            "PREFLIGHT ANALYSIS SUMMARY",
            "=" * 70,
            ""
        ]
        
        # Decision
        if report['can_proceed']:
            lines.append("[OK] READY TO PROCEED")
            lines.append(f"Confidence: {report['confidence'].upper()}")
        else:
            lines.append("[FAIL] CANNOT PROCEED")
            lines.append("Critical issues must be resolved first.")
        lines.append("")

        risk_profile = report.get('risk_profile', {})
        if risk_profile:
            lines.append(
                f"Risk Level: {risk_profile.get('level', 'unknown').upper()} "
                f"(score: {risk_profile.get('score', 0)})"
            )
            if risk_profile.get('drivers'):
                lines.append("Top Risk Drivers:")
                for driver in risk_profile['drivers'][:3]:
                    lines.append(f"  • {driver}")
            if risk_profile.get('rationale'):
                lines.append("Key Rationale:")
                for reason in risk_profile['rationale'][:3]:
                    lines.append(f"  • {reason}")
            lines.append("")
        
        # Blockers
        if report['blockers']:
            lines.append("[!] BLOCKERS:")
            for blocker in report['blockers']:
                lines.append(f"  • {blocker}")
            lines.append("")
        
        # Warnings
        if report['warnings']:
            lines.append(f"  WARNINGS ({len(report['warnings'])}):")
            for warning in report['warnings'][:5]:  # Show first 5
                lines.append(f"  • {warning}")
            if len(report['warnings']) > 5:
                lines.append(f"  ... and {len(report['warnings']) - 5} more")
            lines.append("")
        
        # Recommendations
        if report['recommendations']:
            lines.append("[TIP] RECOMMENDATIONS:")
            for rec in report['recommendations'][:5]:  # Show first 5
                lines.append(f"  • {rec}")
            if len(report['recommendations']) > 5:
                lines.append(f"  ... and {len(report['recommendations']) - 5} more")
            lines.append("")
        
        # Optimized Configuration
        opt = report['optimized_config']
        lines.extend([
            "  OPTIMIZED CONFIGURATION:",
            f"  Device: {opt['device']}",
            f"  Precision: {opt['precision']}",
            f"  Batch Size: {opt['batch_size']}",
            f"  Workers: {opt['num_workers']}",
            f"  Use AMP: {opt['use_amp']}",
            ""
        ])
        
        if opt.get('distillation_strategy'):
            strategy = opt['distillation_strategy']
            lines.append(f"  Recommended Strategy: {strategy.get('primary_method', 'unknown')}")
            lines.append("")
        if opt.get('recommended_preset'):
            preset = opt['recommended_preset']
            lines.append(f"  Suggested Preset: {preset.get('name')}" )
            rationale = preset.get('rationale', [])
            if rationale:
                lines.append("    Rationale:")
                for reason in rationale[:2]:
                    lines.append(f"      - {reason}")
            lines.append("")
        
        # Summary
        model = report['model_analysis']
        data = report['data_analysis']
        resources = report['resource_profile']
        
        lines.extend([
            "[INFO] ANALYSIS SUMMARY:",
            f"  Teacher: {model.get('teacher', {}).get('type', 'unknown')} "
            f"({model.get('teacher', {}).get('total_params', 0) / 1e6:.1f}M params)",
            f"  Student: {model.get('student', {}).get('type', 'unknown')} "
            f"({model.get('student', {}).get('total_params', 0) / 1e6:.1f}M params)",
            f"  Compression: {model.get('compression_ratio', 0):.1f}x",
            f"  Dataset: {data.get('statistics', {}).get('num_samples', 'unknown')} samples",
            f"  Task: {data.get('task_type', 'unknown')}",
            f"  Device: {resources['devices']['primary']}",
            ""
        ])
        
        playbook = report.get('playbook', {})
        if playbook.get('priority_actions'):
            lines.append(">>> NEXT STEPS:")
            for action in playbook['priority_actions'][:3]:
                lines.append(f"  • {action}")
            lines.append("")

        lines.append("=" * 70)
        
        return "\n".join(lines)

    def _compute_risk_profile(self) -> Dict[str, Any]:
        """Aggregate blockers, warnings, and critical metrics into a risk score."""
        blockers = self.results.get('decision', {}).get('blockers', [])
        warnings = self.results.get('decision', {}).get('warnings', [])
        data_stats = self.results.get('data', {}).get('statistics', {}) or {}
        compression = self.results.get('model', {}).get('compression_ratio')
        train_cfg = self.config.get('train', {}) if hasattr(self, 'config') else {}
        guard_cfg = train_cfg.get('overfit_guard', True)
        if isinstance(guard_cfg, dict):
            guard_enabled = guard_cfg.get('enabled', True)
        else:
            guard_enabled = bool(guard_cfg)
        teacher_training_enabled = bool(train_cfg.get('train_teacher', False))
        mitigation_cfg = train_cfg.get('overfit_mitigation', True)
        if isinstance(mitigation_cfg, dict):
            mitigation_enabled = mitigation_cfg.get('enabled', True)
        elif mitigation_cfg in (False, None):
            mitigation_enabled = False
        else:
            mitigation_enabled = True

        risk_score = 100
        rationale: List[str] = []
        metrics: Dict[str, Any] = {}

        risk_score -= len(blockers) * 35
        if blockers:
            rationale.append(f"{len(blockers)} blocker(s) must be resolved before launch.")

        warning_penalty = min(len(warnings), 10) * 5
        risk_score -= warning_penalty
        if warnings:
            rationale.append(f"{len(warnings)} warnings identified that could impact quality.")

        if isinstance(compression, (float, int)):
            metrics['compression_ratio'] = compression
            if compression > 10:
                risk_score -= 10
                rationale.append("Compression ratio above 10x may degrade accuracy.")

        num_samples = data_stats.get('num_samples')
        if isinstance(num_samples, int):
            metrics['num_samples'] = num_samples
            if num_samples < 500:
                risk_score -= 10
                rationale.append("Dataset contains fewer than 500 samples; overfitting risk increases.")

        metrics['overfit_guard_enabled'] = guard_enabled
        if not guard_enabled:
            risk_score -= 8
            rationale.append("Overfit guard disabled; enable automatic halt to contain overfitting.")

        metrics['adaptive_mitigation_enabled'] = mitigation_enabled
        if not mitigation_enabled:
            risk_score -= 5
            rationale.append("Adaptive overfit mitigation disabled; guard cannot attempt regularization before stopping.")

        metrics['teacher_warmup_enabled'] = teacher_training_enabled
        if not teacher_training_enabled:
            risk_score -= 6
            rationale.append("Teacher fine-tuning disabled; teacher quality may bottleneck distillation.")

        imbalance = data_stats.get('imbalance_ratio')
        if isinstance(imbalance, (float, int)):
            metrics['imbalance_ratio'] = imbalance
            if imbalance > 5:
                risk_score -= 8
                rationale.append(f"Class imbalance ratio {imbalance:.1f}:1 requires mitigation.")

        risk_score = max(0, min(100, risk_score))

        if risk_score >= 70:
            level = 'low'
        elif risk_score >= 40:
            level = 'medium'
        else:
            level = 'high'

        drivers = blockers + warnings[:5]

        if not rationale:
            rationale.append("No major risks detected; proceed with standard monitoring.")

        return {
            'score': risk_score,
            'level': level,
            'drivers': drivers,
            'rationale': rationale,
            'critical_metrics': metrics,
        }

    def _generate_user_playbook(self) -> Dict[str, List[str]]:
        """Produce an action-oriented playbook for non-technical stakeholders."""
        decision = self.results.get('decision', {})
        optimization = self.results.get('optimization', {})
        data_report = self.results.get('data', {})
        model_report = self.results.get('model', {})

        actions: List[str] = []  # type: ignore[var-annotated]
        monitor: List[str] = []  # type: ignore[var-annotated]
        ready: List[str] = []  # type: ignore[var-annotated]

        if not decision.get('can_proceed', False):
            actions.extend(decision.get('blockers', []) or ["Resolve blockers before proceeding."])
        else:
            ready.append("All systems go. Initiate distillation when convenient.")

        if data_report.get('warnings'):
            monitor.append("Dataset warnings detected; review imbalance and preprocessing suggestions.")
        if model_report.get('compatibility', {}).get('warnings'):
            monitor.append("Model compatibility warnings present; keep adaptive features enabled.")
        if optimization.get('recommended_preset'):
            ready.append(
                f"Apply preset '{optimization['recommended_preset']['name']}' for automated setup."
            )

        if optimization.get('use_amp'):
            ready.append("Enable mixed precision (AMP) to speed up training on GPU.")

        return {
            'priority_actions': actions,
            'monitor': monitor,
            'ready_to_launch': ready,
        }
    
    def save_report(
        self,
        report: Optional[Dict[str, Any]] = None,
        format: str = 'json'
    ) -> Path:
        """
        Save report to file.
        
        Args:
            report: Report dictionary (uses last run if not provided)
            format: Output format ('json', 'yaml', or 'txt')
            
        Returns:
            Path to saved report
        """
        if report is None:
            report = self._generate_comprehensive_report()
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"preflight_report_{timestamp}.{format}"
        filepath = self.output_dir / filename
        
        # Save based on format
        if format == 'json':
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
        
        elif format == 'yaml':
            with open(filepath, 'w') as f:
                yaml.dump(report, f, default_flow_style=False)
        
        elif format == 'txt':
            with open(filepath, 'w') as f:
                f.write(self._format_final_report(report))
                f.write("\n\n")
                
                f.write("=" * 70 + "\n")
                f.write("RISK PROFILE\n")
                f.write("=" * 70 + "\n")
                risk = report.get('risk_profile', {})
                if risk:
                    f.write(f"Level: {risk.get('level', 'unknown')} (score {risk.get('score', 0)})\n")
                    for reason in risk.get('rationale', []):
                        f.write(f" - {reason}\n")
                    metrics = risk.get('critical_metrics', {})
                    if metrics:
                        f.write("Critical Metrics:\n")
                        for name, value in metrics.items():
                            f.write(f"   {name}: {value}\n")
                else:
                    f.write("No risk profile available.\n")
                f.write("\n")

                f.write("=" * 70 + "\n")
                f.write("ACTION PLAYBOOK\n")
                f.write("=" * 70 + "\n")
                playbook = report.get('playbook', {})
                for section_name, steps in playbook.items():
                    title = section_name.replace('_', ' ').title()
                    f.write(f"{title}:\n")
                    if steps:
                        for step in steps:
                            f.write(f" - {step}\n")
                    else:
                        f.write(" (no actions)\n")
                    f.write("\n")

                # Add detailed reports
                f.write("=" * 70 + "\n")
                f.write("DETAILED MODEL ANALYSIS\n")
                f.write("=" * 70 + "\n")
                f.write(self.model_inspector.generate_report())
                f.write("\n\n")
                
                f.write("=" * 70 + "\n")
                f.write("DETAILED DATA ANALYSIS\n")
                f.write("=" * 70 + "\n")
                f.write(self.data_inspector.generate_report())
                f.write("\n\n")
                
                f.write("=" * 70 + "\n")
                f.write("DETAILED RESOURCE PROFILE\n")
                f.write("=" * 70 + "\n")
                f.write(self.resource_probe.generate_report())
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Report saved to: {filepath}")
        return filepath
    
    def update_config(
        self,
        config: Optional[Dict] = None,
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Update configuration with optimized settings.
        
        Args:
            config: Config to update (uses self.config if not provided)
            save_path: Path to save updated config
            
        Returns:
            Updated configuration
        """
        if config is None:
            config = self.config.copy()
        
        opt = self.results['optimization']
        
        # Update training settings
        if 'training' not in config:
            config['training'] = {}
        
        config['training']['device'] = opt['device']
        config['training']['precision'] = opt['precision']
        config['training']['use_amp'] = opt['use_amp']
        
        # Update data settings
        if 'data' not in config:
            config['data'] = {}
        
        config['data']['batch_size'] = opt['batch_size']
        config['data']['num_workers'] = opt['num_workers']
        config['data']['pin_memory'] = opt['pin_memory']
        
        # Update distillation settings
        if opt.get('distillation_strategy'):
            if 'distillation' not in config:
                config['distillation'] = {}
            
            strategy = opt['distillation_strategy']
            config['distillation']['method'] = strategy.get('primary_method', 'kd_hinton')
            
            if opt.get('layer_mapping'):
                config['distillation']['layer_mapping'] = opt['layer_mapping']
        
        # Save if path provided
        if save_path:
            save_path_obj = Path(save_path)
            save_path_obj.parent.mkdir(parents=True, exist_ok=True)
            
            with open(save_path_obj, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            logger.info(f"Updated config saved to: {save_path_obj}")
        return config


# Convenience function
def run_preflight_check(
    teacher_model: Optional[nn.Module] = None,
    student_model: Optional[nn.Module] = None,
    dataset: Optional[Dataset] = None,
    config: Optional[Dict] = None,
    save_report: bool = True,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to run complete preflight check.
    
    Args:
        teacher_model: Teacher model
        student_model: Student model
        dataset: Training dataset
        config: Configuration dictionary
        save_report: Whether to save report to file
        output_dir: Directory for reports
        
    Returns:
        Comprehensive analysis report
    """
    analyzer = PreflightAnalyzer(
        teacher_model=teacher_model,
        student_model=student_model,
        dataset=dataset,
        config=config,
        output_dir=output_dir
    )
    
    report = analyzer.run_preflight(verbose=True)
    
    if save_report:
        analyzer.save_report(report, format='json')
        analyzer.save_report(report, format='txt')
    
    return report


def validate_config_only(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate configuration before model loading (Phase 1.1).
    
    Use this as the first step to catch config errors early.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Validation report with errors and warnings
        
    Example:
        >>> from zynthe.core.preflight.analyser import validate_config_only
        >>> validation = validate_config_only(cfg_manager.resolved_config)
        >>> if not validation['is_valid']:
        >>>     print("Config errors:", validation['errors'])
        >>>     exit(1)
    """
    analyzer = PreflightAnalyzer(config=config)
    validation = analyzer.validate_config()
    
    logger.info("=" * 70)
    logger.info("CONFIG VALIDATION")
    logger.info("=" * 70)
    print()
    logger.info(analyzer._format_config_validation(validation))
    print()
    logger.info("=" * 70)
    return validation
