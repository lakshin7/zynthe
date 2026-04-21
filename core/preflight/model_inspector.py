"""
Model Inspector - Architecture Analysis & Compatibility Checking
=================================================================

Performs deep inspection of teacher and student models:
1. Model type detection (Vision/NLP/Multimodal/Audio/Video)
2. Architecture family classification (CNN/Transformer/Hybrid)
3. Parameter counting and compression ratio calculation
4. Layer structure analysis and auto-mapping
5. Feature dimension extraction
6. Compatibility verification

Supports:
- Vision: ResNet, ViT, DeiT, Swin, ConvNeXt, EfficientNet
- NLP: BERT, RoBERTa, GPT, T5, DistilBERT, ALBERT
- Multimodal: CLIP, BLIP, Qwen-VL, LLaVA
- Audio: Wav2Vec2, Whisper, HuBERT
- Video: TimeSformer, VideoMAE, ViViT
"""

from typing import Dict, List, Optional, Any
import torch.nn as nn
from collections import defaultdict
import numpy as np


class ModelInspector:
    """
    Comprehensive model architecture inspector.
    
    Analyzes teacher and student models to extract:
    - Model type and architecture family
    - Parameter counts and compression ratios
    - Layer structures and mappings
    - Feature dimensions
    - Compatibility status
    """
    
    def __init__(self, teacher: Optional[nn.Module] = None, student: Optional[nn.Module] = None):
        """
        Initialize model inspector.
        
        Args:
            teacher: Teacher model (optional)
            student: Student model (optional)
        """
        self.teacher = teacher
        self.student = student
        self.teacher_info = {}
        self.student_info = {}
        self.compatibility = {}
    
    def inspect(self) -> Dict[str, Any]:
        """
        Run full inspection on both models.
        
        Returns:
            Dictionary with comprehensive analysis results
        """
        report = {}
        
        signatures: Dict[str, Any] = {}

        if self.teacher is not None:
            self.teacher_info = self._inspect_model(self.teacher, "teacher")
            report['teacher'] = self.teacher_info
            signatures['teacher'] = self.teacher_info.get('signature')
        
        if self.student is not None:
            self.student_info = self._inspect_model(self.student, "student")
            report['student'] = self.student_info
            signatures['student'] = self.student_info.get('signature')
        
        if self.teacher is not None and self.student is not None:
            self.compatibility = self._check_compatibility()
            report['compatibility'] = self.compatibility
            report['compression_ratio'] = self._compute_compression_ratio()
            report['recommended_strategy'] = self._recommend_distillation_strategy()
            layer_mapping = self._auto_map_layers()
            report['layer_mapping'] = layer_mapping
            report['distillation_difficulty'] = self._assess_distillation_difficulty(layer_mapping)
            report['preset_alignment'] = self._derive_preset_alignment()

        if signatures:
            report['signatures'] = signatures
        
        return report
    
    def _inspect_model(self, model: nn.Module, role: str) -> Dict[str, Any]:
        """
        Inspect a single model.
        
        Args:
            model: PyTorch model
            role: "teacher" or "student"
            
        Returns:
            Dictionary with model information
        """
        info = {
            'role': role,
            'name': model.__class__.__name__,
            'type': self._detect_model_type(model),
            'architecture_family': self._detect_architecture_family(model),
            'total_params': self._count_parameters(model),
            'trainable_params': self._count_parameters(model, only_trainable=True),
            'layer_structure': self._analyze_layer_structure(model),
            'feature_dimensions': self._extract_feature_dimensions(model),
            'attention_layers': self._count_attention_layers(model),
            'has_embeddings': self._has_embeddings(model),
            'output_shape': self._infer_output_shape(model),
            'depth': self._compute_model_depth(model),
            'width': self._compute_model_width(model),
        }

        info['detection_confidence'] = self._compute_detection_confidence(info)
        info['capabilities'] = self._derive_model_capabilities(model)
        info['signature'] = self._build_model_signature(info)
        
        return info
    
    def _detect_model_type(self, model: nn.Module) -> str:
        """
        Detect model type from architecture.
        
        Returns:
            Model type: vision, nlp, multimodal, audio, video, unknown
        """
        name = model.__class__.__name__.lower()
        
        # NLP models
        nlp_keywords = ['bert', 'roberta', 'gpt', 'gpt2', 'gpt-', 't5', 'bart', 
                        'albert', 'electra', 'deberta', 'xlnet', 'xlm', 'distilbert',
                        'longformer', 'reformer', 'funnel', 'layoutlm']
        if any(kw in name for kw in nlp_keywords):
            return 'nlp'
        
        # Vision models
        vision_keywords = ['resnet', 'vit', 'deit', 'swin', 'convnext', 'efficientnet',
                          'mobilenet', 'densenet', 'inception', 'vgg', 'alexnet',
                          'resnext', 'regnet', 'shufflenet', 'squeezenet']
        if any(kw in name for kw in vision_keywords):
            return 'vision'
        
        # Multimodal models
        multimodal_keywords = ['clip', 'blip', 'qwen', 'llava', 'flamingo', 
                               'coca', 'align', 'florence', 'beit3']
        if any(kw in name for kw in multimodal_keywords):
            return 'multimodal'
        
        # Audio models
        audio_keywords = ['wav2vec', 'hubert', 'whisper', 'wavlm', 'audio']
        if any(kw in name for kw in audio_keywords):
            return 'audio'
        
        # Video models
        video_keywords = ['timesformer', 'videomae', 'vivit', 'video', 'i3d', 'slowfast']
        if any(kw in name for kw in video_keywords):
            return 'video'
        
        # Fallback: check module structure
        has_conv = any(isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)) for m in model.modules())
        has_attention = any('attention' in m.__class__.__name__.lower() for m in model.modules())
        has_embedding = any(isinstance(m, nn.Embedding) for m in model.modules())
        
        if has_conv and not has_embedding:
            return 'vision'
        elif has_embedding and has_attention:
            return 'nlp'
        elif has_conv and has_embedding:
            return 'multimodal'
        
        return 'unknown'
    
    def _detect_architecture_family(self, model: nn.Module) -> str:
        """
        Detect architecture family.
        
        Returns:
            Family: cnn, transformer, hybrid, rnn, unknown
        """
        # Count module types
        module_counts = defaultdict(int)
        for module in model.modules():
            module_type = module.__class__.__name__.lower()
            
            if 'conv' in module_type:
                module_counts['conv'] += 1
            elif 'attention' in module_type or 'transformer' in module_type:
                module_counts['attention'] += 1
            elif 'lstm' in module_type or 'gru' in module_type or 'rnn' in module_type:
                module_counts['rnn'] += 1
        
        # Classify based on dominant module type
        if module_counts['attention'] > 0 and module_counts['conv'] > 0:
            return 'hybrid'
        elif module_counts['attention'] > module_counts['conv']:
            return 'transformer'
        elif module_counts['conv'] > 0:
            return 'cnn'
        elif module_counts['rnn'] > 0:
            return 'rnn'
        
        return 'unknown'
    
    def _count_parameters(self, model: nn.Module, only_trainable: bool = False) -> int:
        """Count total or trainable parameters."""
        if only_trainable:
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in model.parameters())
    
    def _analyze_layer_structure(self, model: nn.Module) -> Dict[str, Any]:
        """
        Analyze layer structure.
        
        Returns:
            Dictionary with layer counts by type
        """
        structure = defaultdict(int)
        layer_names = []
        
        for name, module in model.named_modules():
            module_type = module.__class__.__name__
            structure[module_type] += 1
            
            # Track important layers
            if any(kw in module_type.lower() for kw in ['conv', 'linear', 'attention', 'layer']):
                if len(name) > 0:  # Skip root module
                    layer_names.append(name)
        
        return {
            'module_counts': dict(structure),
            'total_modules': len(list(model.modules())),
            'named_layers': layer_names[:50],  # First 50 layers
            'total_named_layers': len(layer_names)
        }
    
    def _extract_feature_dimensions(self, model: nn.Module) -> Dict[str, List[int]]:
        """
        Extract feature dimensions from key layers.
        
        Returns:
            Dictionary mapping layer names to dimensions
        """
        dimensions = {}
        
        for name, module in model.named_modules():
            # Extract dimensions from Linear layers
            if isinstance(module, nn.Linear):
                dimensions[name] = [module.in_features, module.out_features]
            
            # Extract dimensions from Conv layers
            elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                dimensions[name] = [module.in_channels, module.out_channels]
            
            # Extract dimensions from Embedding layers
            elif isinstance(module, nn.Embedding):
                dimensions[name] = [module.num_embeddings, module.embedding_dim]
        
        return dimensions
    
    def _count_attention_layers(self, model: nn.Module) -> int:
        """Count number of attention/transformer layers."""
        count = 0
        for module in model.modules():
            module_name = module.__class__.__name__.lower()
            if 'attention' in module_name or 'transformer' in module_name:
                count += 1
        return count
    
    def _has_embeddings(self, model: nn.Module) -> bool:
        """Check if model has embedding layers."""
        for module in model.modules():
            if isinstance(module, nn.Embedding):
                return True
        return False
    
    def _infer_output_shape(self, model: nn.Module) -> Optional[List[int]]:
        """
        Infer output shape by checking final layers.
        
        Returns:
            Output dimensions if detectable
        """
        # Find last Linear or Conv layer
        last_linear = None
        last_conv = None
        
        for module in model.modules():
            if isinstance(module, nn.Linear):
                last_linear = module
            elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                last_conv = module
        
        if last_linear is not None:
            return [last_linear.out_features]
        elif last_conv is not None:
            return [last_conv.out_channels]
        
        return None
    
    def _compute_model_depth(self, model: nn.Module) -> int:
        """
        Compute model depth (number of sequential layers).
        
        Returns:
            Approximate depth
        """
        # Count Sequential, ModuleList, and named layers with numeric indices
        depth = 0
        
        for name, module in model.named_modules():
            # Look for layer patterns like "layer.0", "encoder.layer.5", etc.
            parts = name.split('.')
            numeric_parts = [p for p in parts if p.isdigit()]
            if numeric_parts:
                max_idx = max(int(p) for p in numeric_parts)
                depth = max(depth, max_idx + 1)
        
        return depth if depth > 0 else len(list(model.children()))
    
    def _compute_model_width(self, model: nn.Module) -> int:
        """
        Compute model width (typical hidden dimension).
        
        Returns:
            Average hidden dimension
        """
        dimensions = []
        
        for module in model.modules():
            if isinstance(module, nn.Linear):
                dimensions.append(module.in_features)
                dimensions.append(module.out_features)
            elif isinstance(module, (nn.Conv2d, nn.Conv1d)):
                dimensions.append(module.out_channels)
        
        return int(np.median(dimensions)) if dimensions else 0

    def _compute_detection_confidence(self, info: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate confidence levels for detected attributes."""
        confidence_scale = {'low': 1, 'medium': 2, 'high': 3}

        def _level(has_high: bool, has_low: bool) -> str:
            if has_high:
                return 'high'
            if has_low:
                return 'low'
            return 'medium'

        layer_stats = info.get('layer_structure', {})
        total_named = layer_stats.get('total_named_layers', 0) or 0
        module_counts = layer_stats.get('module_counts', {}) or {}

        type_level = _level(info.get('type') not in {'unknown', None}, total_named < 3)
        arch_count = module_counts.get('attention', 0) + module_counts.get('conv', 0)
        arch_level = _level(info.get('architecture_family') not in {'unknown', None}, arch_count == 0)
        output_level = _level(bool(info.get('output_shape')), info.get('output_shape') is None)
        depth_level = _level(bool(info.get('depth')), info.get('depth', 0) == 0)

        overall_numeric = sum(
            confidence_scale[level]
            for level in [type_level, arch_level, output_level, depth_level]
        ) / 4

        if overall_numeric >= 2.5:
            overall = 'high'
        elif overall_numeric >= 1.75:
            overall = 'medium'
        else:
            overall = 'low'

        return {
            'type': type_level,
            'architecture': arch_level,
            'output_shape': output_level,
            'depth': depth_level,
            'overall': overall,
        }

    def _derive_model_capabilities(self, model: nn.Module) -> Dict[str, Any]:
        """Infer optional capabilities from module structure and config."""
        module_names = {name.lower() for name, _ in model.named_modules() if name}
        config = getattr(model, 'config', None)

        supports_adapters = any('adapter' in name for name in module_names) or hasattr(model, 'add_adapter')
        supports_lora = any('lora' in name for name in module_names) or hasattr(model, 'lora_layers')
        supports_prefix_tuning = any('prefix' in name for name in module_names)
        gradient_checkpointing = hasattr(model, 'gradient_checkpointing_enable')
        quantization_ready = bool(
            hasattr(model, 'quantization_config') or any('quant' in name for name in module_names)
        )
        sequence_limit = None
        if config is not None:
            sequence_limit = getattr(config, 'max_position_embeddings', None)

        notes: List[str] = []
        if supports_adapters:
            notes.append('Adapter-friendly modules detected.')
        if supports_lora:
            notes.append('LoRA hooks present or supported.')
        if quantization_ready:
            notes.append('Quantization-specific attributes observed.')
        if gradient_checkpointing:
            notes.append('Gradient checkpointing API available.')

        return {
            'supports_adapters': supports_adapters,
            'supports_lora': supports_lora,
            'supports_prefix_tuning': supports_prefix_tuning,
            'gradient_checkpointing': gradient_checkpointing,
            'quantization_ready': quantization_ready,
            'sequence_limit': sequence_limit,
            'notes': notes,
        }

    def _build_model_signature(self, info: Dict[str, Any]) -> Dict[str, Any]:
        """Create a compact signature for change tracking."""
        return {
            'name': info.get('name'),
            'type': info.get('type'),
            'architecture': info.get('architecture_family'),
            'total_params': info.get('total_params'),
            'trainable_params': info.get('trainable_params'),
            'depth': info.get('depth'),
            'width': info.get('width'),
            'attention_layers': info.get('attention_layers'),
        }
    
    def _check_compatibility(self) -> Dict[str, Any]:
        """
        Check teacher-student compatibility.
        
        Returns:
            Compatibility report
        """
        compat = {
            'is_compatible': True,
            'warnings': [],
            'errors': [],
            'recommendations': []
        }
        
        # Check model types
        t_type = self.teacher_info.get('type', 'unknown')
        s_type = self.student_info.get('type', 'unknown')
        
        if t_type != s_type and t_type != 'unknown' and s_type != 'unknown':
            compat['warnings'].append(
                f"Different model types: teacher={t_type}, student={s_type}. "
                f"Cross-domain distillation requires careful configuration."
            )
        
        # Check architecture families
        t_arch = self.teacher_info.get('architecture_family', 'unknown')
        s_arch = self.student_info.get('architecture_family', 'unknown')
        
        if t_arch != s_arch:
            compat['warnings'].append(
                f"Different architectures: teacher={t_arch}, student={s_arch}. "
                f"Feature distillation may require adaptive layers."
            )
            compat['recommendations'].append(
                "Enable auto-adaptive feature matching (1x1 conv projections)"
            )
        
        # Check compression ratio
        ratio = self._compute_compression_ratio()
        if ratio > 10:
            compat['warnings'].append(
                f"High compression ratio ({ratio:.1f}x). "
                f"Consider multi-stage distillation or intermediate student."
            )
            compat['recommendations'].append(
                "Use progressive distillation with hint learning"
            )
        elif ratio < 1.5:
            compat['warnings'].append(
                f"Low compression ratio ({ratio:.1f}x). "
                f"Limited efficiency gains expected."
            )
        
        # Check output shapes
        t_out = self.teacher_info.get('output_shape')
        s_out = self.student_info.get('output_shape')
        
        if t_out and s_out and t_out != s_out:
            compat['errors'].append(
                f"Output shape mismatch: teacher={t_out}, student={s_out}. "
                f"Cannot perform logit distillation."
            )
            compat['is_compatible'] = False
            compat['recommendations'].append(
                "Use feature distillation only, or add projection layer"
            )
        
        return compat
    
    def _compute_compression_ratio(self) -> float:
        """
        Compute teacher-to-student compression ratio.
        
        Returns:
            Compression ratio (teacher_params / student_params)
        """
        t_params = self.teacher_info.get('total_params', 1)
        s_params = self.student_info.get('total_params', 1)
        
        return t_params / max(s_params, 1)
    
    def _recommend_distillation_strategy(self) -> Dict[str, Any]:
        """
        Recommend distillation strategy based on analysis.
        
        Returns:
            Dictionary with recommended strategy
        """
        ratio = self._compute_compression_ratio()
        self.teacher_info.get('type', 'unknown')
        t_arch = self.teacher_info.get('architecture_family', 'unknown')
        s_arch = self.student_info.get('architecture_family', 'unknown')
        
        strategy = {
            'primary_method': 'logit',
            'additional_methods': [],
            'hint_learning': False,
            'adaptive_features': False,
            'temperature': 4.0,
            'alpha': 0.7
        }
        
        # High compression (>8x): Use all techniques
        if ratio > 8:
            strategy['primary_method'] = 'multi_stage'
            strategy['additional_methods'] = ['logit', 'feature', 'attention', 'hint']
            strategy['hint_learning'] = True
            strategy['adaptive_features'] = True
            strategy['temperature'] = 6.0
            strategy['alpha'] = 0.8
        
        # Medium compression (3-8x): Logit + Feature
        elif 3 <= ratio <= 8:
            strategy['primary_method'] = 'logit'
            strategy['additional_methods'] = ['feature']
            strategy['hint_learning'] = ratio > 5
            strategy['temperature'] = 4.0
            strategy['alpha'] = 0.7
        
        # Low compression (<3x): Feature only or simple logit
        else:
            strategy['primary_method'] = 'feature'
            strategy['temperature'] = 2.0
            strategy['alpha'] = 0.5
        
        # Cross-architecture requires adaptive features
        if t_arch != s_arch:
            strategy['adaptive_features'] = True
            strategy['additional_methods'].append('adaptive_hint')
        
        # Transformer models benefit from attention distillation
        if t_arch == 'transformer' and s_arch == 'transformer':
            if 'attention' not in strategy['additional_methods']:
                strategy['additional_methods'].append('attention')
        
        return strategy

    def _assess_distillation_difficulty(
        self,
        layer_mapping: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate a heuristic complexity score for the distillation plan."""
        if not self.teacher_info or not self.student_info:
            return {
                'score': None,
                'level': 'unknown',
                'drivers': ['Insufficient information to assess difficulty.'],
                'recommended_actions': []
            }

        score = 100
        drivers: List[str] = []
        actions: List[str] = []

        ratio = self._compute_compression_ratio()
        if ratio > 12:
            score -= 30
            drivers.append(f'Extreme compression target detected ({ratio:.1f}x).')
            actions.append('Consider multi-stage distillation or intermediate students.')
        elif ratio > 8:
            score -= 18
            drivers.append(f'High compression target ({ratio:.1f}x).')
        elif ratio < 1.5:
            score -= 5
            drivers.append('Compression ratio is low; efficiency gains may be limited.')

        teacher_depth = max(self.teacher_info.get('depth') or 0, 1)
        student_depth = max(self.student_info.get('depth') or 0, 1)
        depth_gap = teacher_depth / max(student_depth, 1)
        if depth_gap > 2.5:
            score -= 15
            drivers.append('Teacher depth significantly exceeds student depth.')
            actions.append('Add adapter layers or intermediate feature heads for deep teachers.')
        elif depth_gap < 0.7:
            score -= 5
            drivers.append('Student deeper than teacher; verify architecture alignment.')

        teacher_type = self.teacher_info.get('type')
        student_type = self.student_info.get('type')
        if teacher_type and student_type and teacher_type != student_type:
            score -= 12
            drivers.append(f'Type mismatch (teacher={teacher_type}, student={student_type}).')
            actions.append('Enable cross-domain feature adapters or revisit student selection.')

        t_attention = self.teacher_info.get('attention_layers', 0) or 0
        s_attention = self.student_info.get('attention_layers', 0) or 0
        if t_attention > 0 and s_attention == 0:
            score -= 12
            drivers.append('Teacher uses attention blocks but student does not.')
            actions.append('Add attention heads or rely on feature distillation only.')

        compat = self.compatibility or {}
        errors = compat.get('errors', [])
        warnings_list = compat.get('warnings', [])
        if errors:
            score -= 35
            drivers.append('Compatibility errors present. Immediate mitigation required.')
            actions.extend(compat.get('recommendations', []))
        if warnings_list:
            warning_penalty = min(len(warnings_list), 5) * 4
            score -= warning_penalty
            drivers.append(f'{len(warnings_list)} compatibility warnings detected.')

        if layer_mapping:
            quality = layer_mapping.get('quality')
            if quality == 'low':
                score -= 12
                drivers.append('Layer mapping coverage is low.')
                actions.append('Manually map key transformer blocks or attention heads.')
            elif quality == 'medium':
                score -= 6
                drivers.append('Layer mapping coverage is partial.')

        score = max(0, min(100, score))

        if score >= 70:
            level = 'manageable'
        elif score >= 40:
            level = 'challenging'
        else:
            level = 'high_risk'

        # Deduplicate actions while preserving order
        seen = set()
        unique_actions = []
        for action in actions:
            if action and action not in seen:
                seen.add(action)
                unique_actions.append(action)

        metrics = {
            'compression_ratio': ratio,
            'depth_ratio': round(depth_gap, 3),
            'attention_gap': max(t_attention - s_attention, 0)
        }

        return {
            'score': score,
            'level': level,
            'drivers': drivers,
            'recommended_actions': unique_actions,
            'metrics': metrics
        }

    def _derive_preset_alignment(self) -> Dict[str, Any]:
        """Suggest distillation presets informed by model pairing."""
        if not self.teacher_info or not self.student_info:
            return {
                'primary': None,
                'alternatives': [],
                'signals': ['Preset selection requires both teacher and student models.']
            }

        ratio = self._compute_compression_ratio()
        teacher_type = self.teacher_info.get('type', 'unknown')
        teacher_arch = self.teacher_info.get('architecture_family', 'unknown')
        student_arch = self.student_info.get('architecture_family', 'unknown')

        signals: List[str] = []
        candidate = 'quick_start'
        confidence = 'medium'

        if ratio > 8:
            candidate = 'compression_max'
            confidence = 'high'
            signals.append('High compression ratio suggests aggressive compression preset.')
        elif teacher_arch == 'transformer' and teacher_type == 'nlp':
            candidate = 'balanced'
            confidence = 'high'
            signals.append('Transformer NLP pairing detected; balanced preset recommended.')
        elif teacher_type in {'vision', 'video'}:
            candidate = 'vision_transformer'
            signals.append('Vision architecture detected; use vision-tuned preset.')
        elif teacher_type == 'audio':
            candidate = 'audio_specialized'
            signals.append('Audio model detected; prefer audio-focused preset.')
        elif ratio < 2:
            candidate = 'quick_start'
            signals.append('Low compression target; quick start preset is sufficient.')

        if teacher_arch != student_arch:
            signals.append('Architecture mismatch; ensure preset enables adaptive features.')

        alternatives: List[Dict[str, Any]] = []
        if candidate != 'balanced':
            alternatives.append({'name': 'balanced', 'confidence': 'medium'})
        if candidate != 'quick_start':
            alternatives.append({'name': 'quick_start', 'confidence': 'medium'})

        return {
            'primary': {
                'name': candidate,
                'confidence': confidence,
                'signals': signals
            },
            'alternatives': alternatives,
            'signals': signals,
        }
    
    def _auto_map_layers(self) -> Dict[str, Any]:
        """
        Automatically map teacher layers to student layers.
        
        Returns:
            Dictionary with layer mappings for different distillation types
        """
        t_layers = self.teacher_info.get('layer_structure', {}).get('named_layers', [])
        s_layers = self.student_info.get('layer_structure', {}).get('named_layers', [])
        
        if not t_layers or not s_layers:
            return {'mappings': [], 'confidence': 'low'}
        
        # Filter to get key layers only
        t_key_layers = self._filter_key_layers(t_layers)
        s_key_layers = self._filter_key_layers(s_layers)
        
        # Create mapping based on relative position
        mappings = []
        n_teacher = len(t_key_layers)
        n_student = len(s_key_layers)
        
        if n_teacher > 0 and n_student > 0:
            # Map student layers to proportional teacher layers
            for i, s_layer in enumerate(s_key_layers):
                # Map to proportional position in teacher
                t_idx = int(i * (n_teacher - 1) / max(n_student - 1, 1))
                t_layer = t_key_layers[t_idx]
                
                mappings.append({
                    'teacher': t_layer,
                    'student': s_layer,
                    'confidence': 'high' if abs(i / n_student - t_idx / n_teacher) < 0.2 else 'medium'
                })
        
        teacher_mapped = {m['teacher'] for m in mappings}
        student_mapped = {m['student'] for m in mappings}
        teacher_coverage = len(teacher_mapped) / max(n_teacher, 1)
        student_coverage = len(student_mapped) / max(n_student, 1)

        min_coverage = min(teacher_coverage, student_coverage)
        if min_coverage >= 0.75:
            quality = 'high'
        elif min_coverage >= 0.45:
            quality = 'medium'
        else:
            quality = 'low'

        return {
            'mappings': mappings,
            'teacher_depth': n_teacher,
            'student_depth': n_student,
            'total_mappings': len(mappings),
            'coverage': {
                'teacher': round(teacher_coverage, 3),
                'student': round(student_coverage, 3)
            },
            'quality': quality,
            'unmapped_teacher_layers': max(n_teacher - len(teacher_mapped), 0),
            'unmapped_student_layers': max(n_student - len(student_mapped), 0)
        }
    
    def _filter_key_layers(self, layer_names: List[str]) -> List[str]:
        """
        Filter to get key layers for mapping.
        
        Args:
            layer_names: List of all layer names
            
        Returns:
            Filtered list of key layers
        """
        key_layers = []
        
        # Prioritize encoder/decoder layers, transformer blocks, etc.
        priority_keywords = ['encoder.layer', 'decoder.layer', 'transformer', 
                            'block', 'stage', 'layer']
        
        for name in layer_names:
            if any(kw in name.lower() for kw in priority_keywords):
                # Avoid duplicate submodules (e.g., keep "layer.0" not "layer.0.attention")
                if name.count('.') <= 3:  # Not too deep
                    key_layers.append(name)
        
        # If no priority layers found, use all layers
        if not key_layers:
            key_layers = layer_names
        
        return key_layers
    
    def generate_report(self) -> str:
        """
        Generate human-readable inspection report.
        
        Returns:
            Formatted report string
        """
        report = self.inspect()
        
        lines = [
            "=" * 70,
            "MODEL INSPECTION REPORT",
            "=" * 70,
            ""
        ]
        
        # Teacher info
        if 'teacher' in report:
            t = report['teacher']
            lines.extend([
                "TEACHER MODEL",
                "-" * 70,
                f"Name: {t['name']}",
                f"Type: {t['type']}",
                f"Architecture: {t['architecture_family']}",
                f"Total Parameters: {t['total_params']:,}",
                f"Trainable Parameters: {t['trainable_params']:,}",
                f"Depth: {t['depth']} layers",
                f"Width: {t['width']} dimensions",
                f"Attention Layers: {t['attention_layers']}",
                f"Has Embeddings: {t['has_embeddings']}",
                f"Detection Confidence: {t['detection_confidence']['overall']}",
                f"Supports Adapters: {'Yes' if t['capabilities']['supports_adapters'] else 'No'}",
                f"Supports LoRA: {'Yes' if t['capabilities']['supports_lora'] else 'No'}",
                ""
            ])
        
        # Student info
        if 'student' in report:
            s = report['student']
            lines.extend([
                "STUDENT MODEL",
                "-" * 70,
                f"Name: {s['name']}",
                f"Type: {s['type']}",
                f"Architecture: {s['architecture_family']}",
                f"Total Parameters: {s['total_params']:,}",
                f"Trainable Parameters: {s['trainable_params']:,}",
                f"Depth: {s['depth']} layers",
                f"Width: {s['width']} dimensions",
                f"Attention Layers: {s['attention_layers']}",
                f"Has Embeddings: {s['has_embeddings']}",
                f"Detection Confidence: {s['detection_confidence']['overall']}",
                f"Supports Adapters: {'Yes' if s['capabilities']['supports_adapters'] else 'No'}",
                f"Supports LoRA: {'Yes' if s['capabilities']['supports_lora'] else 'No'}",
                ""
            ])
        
        # Compatibility
        if 'compatibility' in report:
            compat = report['compatibility']
            lines.extend([
                "COMPATIBILITY ANALYSIS",
                "-" * 70,
                f"Status: {'✓ Compatible' if compat['is_compatible'] else '✗ Incompatible'}",
                f"Compression Ratio: {report['compression_ratio']:.2f}x",
                ""
            ])
            
            if compat['warnings']:
                lines.append("Warnings:")
                for w in compat['warnings']:
                    lines.append(f"  ⚠ {w}")
                lines.append("")
            
            if compat['errors']:
                lines.append("Errors:")
                for e in compat['errors']:
                    lines.append(f"  ✗ {e}")
                lines.append("")
            
            if compat['recommendations']:
                lines.append("Recommendations:")
                for r in compat['recommendations']:
                    lines.append(f"  → {r}")
                lines.append("")

        if 'distillation_difficulty' in report:
            difficulty = report['distillation_difficulty']
            lines.extend([
                "DISTILLATION DIFFICULTY",
                "-" * 70,
                f"Score: {difficulty['score'] if difficulty['score'] is not None else 'n/a'}",
                f"Level: {difficulty['level'].replace('_', ' ').title()}",
                "Drivers:"
            ])
            for driver in difficulty.get('drivers', [])[:5]:
                lines.append(f"  • {driver}")
            if difficulty.get('recommended_actions'):
                lines.append("Recommended Actions:")
                for action in difficulty['recommended_actions'][:5]:
                    lines.append(f"  → {action}")
            metrics = difficulty.get('metrics', {})
            if metrics:
                lines.append("Key Metrics:")
                for key, value in metrics.items():
                    lines.append(f"  - {key}: {value}")
            lines.append("")

        if 'preset_alignment' in report:
            preset = report['preset_alignment']
            primary = preset.get('primary') or {}
            lines.extend([
                "PRESET ALIGNMENT",
                "-" * 70,
                f"Primary Recommendation: {primary.get('name', 'n/a')} (confidence: {primary.get('confidence', 'n/a')})"
            ])
            for signal in primary.get('signals', [])[:4]:
                lines.append(f"  • {signal}")
            if preset.get('alternatives'):
                alt_names = ', '.join(a['name'] for a in preset['alternatives'])
                lines.append(f"Alternatives: {alt_names}")
            lines.append("")
        
        # Recommended strategy
        if 'recommended_strategy' in report:
            strat = report['recommended_strategy']
            lines.extend([
                "RECOMMENDED DISTILLATION STRATEGY",
                "-" * 70,
                f"Primary Method: {strat['primary_method']}",
                f"Additional Methods: {', '.join(strat['additional_methods']) if strat['additional_methods'] else 'None'}",
                f"Hint Learning: {'Yes' if strat['hint_learning'] else 'No'}",
                f"Adaptive Features: {'Yes' if strat['adaptive_features'] else 'No'}",
                f"Temperature: {strat['temperature']}",
                f"Alpha (KD weight): {strat['alpha']}",
                ""
            ])
        
        # Layer mapping
        if 'layer_mapping' in report:
            mapping = report['layer_mapping']
            lines.extend([
                "LAYER MAPPING",
                "-" * 70,
                f"Teacher Depth: {mapping['teacher_depth']}",
                f"Student Depth: {mapping['student_depth']}",
                f"Total Mappings: {mapping['total_mappings']}",
                f"Quality: {mapping.get('quality', 'unknown').title()}",
                f"Teacher Coverage: {mapping.get('coverage', {}).get('teacher', 0):.2f}",
                f"Student Coverage: {mapping.get('coverage', {}).get('student', 0):.2f}",
                ""
            ])
            
            if mapping['mappings']:
                lines.append("Suggested Mappings (first 5):")
                for m in mapping['mappings'][:5]:
                    lines.append(f"  {m['student']} ← {m['teacher']} ({m['confidence']} confidence)")
                lines.append("")
        
        lines.append("=" * 70)
        
        return "\n".join(lines)
