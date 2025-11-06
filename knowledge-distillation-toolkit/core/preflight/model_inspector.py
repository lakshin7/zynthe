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

from typing import Dict, List, Tuple, Optional, Any, Set
import torch
import torch.nn as nn
from collections import OrderedDict, defaultdict
import numpy as np
import warnings


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
        
        if self.teacher is not None:
            self.teacher_info = self._inspect_model(self.teacher, "teacher")
            report['teacher'] = self.teacher_info
        
        if self.student is not None:
            self.student_info = self._inspect_model(self.student, "student")
            report['student'] = self.student_info
        
        if self.teacher is not None and self.student is not None:
            self.compatibility = self._check_compatibility()
            report['compatibility'] = self.compatibility
            report['compression_ratio'] = self._compute_compression_ratio()
            report['recommended_strategy'] = self._recommend_distillation_strategy()
            report['layer_mapping'] = self._auto_map_layers()
        
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
        t_type = self.teacher_info.get('type', 'unknown')
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
        
        return {
            'mappings': mappings,
            'teacher_depth': n_teacher,
            'student_depth': n_student,
            'total_mappings': len(mappings)
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
                ""
            ])
            
            if mapping['mappings']:
                lines.append("Suggested Mappings (first 5):")
                for m in mapping['mappings'][:5]:
                    lines.append(f"  {m['student']} ← {m['teacher']} ({m['confidence']} confidence)")
                lines.append("")
        
        lines.append("=" * 70)
        
        return "\n".join(lines)
