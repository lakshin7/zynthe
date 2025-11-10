"""
Student Sizing Heuristics
=========================

Rules and heuristics for automatically sizing student models based on teacher architecture.

Heuristics:
1. Depth Reduction: Reduce number of layers proportionally
2. Width Reduction: Reduce hidden dimension size
3. Attention Head Scaling: Scale number of attention heads
4. FFN Dimension Scaling: Scale feed-forward network dimension
5. Parameter Count Targeting: Ensure target compression ratio
"""

import math
from typing import Dict, Any, Optional
import logging

LOG = logging.getLogger(__name__)


class StudentSizingHeuristics:
    """
    Heuristic rules for determining student architecture dimensions.
    
    Compression strategies:
    - Conservative (0.6-0.8): Primarily depth reduction
    - Balanced (0.4-0.6): Depth + width reduction
    - Aggressive (0.2-0.4): Depth + width + attention reduction
    """
    
    # Known teacher architectures (from Hugging Face)
    KNOWN_TEACHERS = {
        'bert-base-uncased': {
            'num_layers': 12,
            'hidden_size': 768,
            'num_attention_heads': 12,
            'intermediate_size': 3072,
            'vocab_size': 30522,
            'total_params': 110_000_000,
        },
        'bert-large-uncased': {
            'num_layers': 24,
            'hidden_size': 1024,
            'num_attention_heads': 16,
            'intermediate_size': 4096,
            'vocab_size': 30522,
            'total_params': 336_000_000,
        },
        'roberta-base': {
            'num_layers': 12,
            'hidden_size': 768,
            'num_attention_heads': 12,
            'intermediate_size': 3072,
            'vocab_size': 50265,
            'total_params': 125_000_000,
        },
        'roberta-large': {
            'num_layers': 24,
            'hidden_size': 1024,
            'num_attention_heads': 16,
            'intermediate_size': 4096,
            'vocab_size': 50265,
            'total_params': 355_000_000,
        },
        'albert-base-v2': {
            'num_layers': 12,
            'hidden_size': 768,
            'num_attention_heads': 12,
            'intermediate_size': 3072,
            'vocab_size': 30000,
            'total_params': 12_000_000,  # Parameter sharing
        },
        'distilbert-base-uncased': {
            'num_layers': 6,
            'hidden_size': 768,
            'num_attention_heads': 12,
            'intermediate_size': 3072,
            'vocab_size': 30522,
            'total_params': 66_000_000,
        },
    }
    
    @staticmethod
    def get_teacher_config(teacher_name: str) -> Optional[Dict[str, Any]]:
        """
        Get known teacher configuration.
        
        Args:
            teacher_name: Name of teacher model
            
        Returns:
            Teacher config dict or None if unknown
        """
        # Try exact match
        if teacher_name in StudentSizingHeuristics.KNOWN_TEACHERS:
            return StudentSizingHeuristics.KNOWN_TEACHERS[teacher_name].copy()
        
        # Try partial match (e.g., "bert-base" matches "bert-base-uncased")
        for known_name, config in StudentSizingHeuristics.KNOWN_TEACHERS.items():
            if teacher_name.lower() in known_name.lower() or known_name.lower() in teacher_name.lower():
                LOG.info(f"Matched teacher '{teacher_name}' to known config '{known_name}'")
                return config.copy()
        
        LOG.warning(f"Unknown teacher model: {teacher_name}")
        return None
    
    @staticmethod
    def calculate_student_dimensions(
        teacher_config: Dict[str, Any],
        compression_ratio: float = 0.5,
        strategy: str = 'balanced'
    ) -> Dict[str, Any]:
        """
        Calculate student dimensions based on compression ratio and strategy.
        
        Args:
            teacher_config: Teacher architecture config
            compression_ratio: Target size ratio (0.5 = 50% of teacher)
            strategy: Compression strategy ('conservative', 'balanced', 'aggressive')
            
        Returns:
            Student architecture config
        """
        if strategy == 'conservative':
            return StudentSizingHeuristics._conservative_sizing(teacher_config, compression_ratio)
        elif strategy == 'balanced':
            return StudentSizingHeuristics._balanced_sizing(teacher_config, compression_ratio)
        elif strategy == 'aggressive':
            return StudentSizingHeuristics._aggressive_sizing(teacher_config, compression_ratio)
        else:
            LOG.warning(f"Unknown strategy '{strategy}', using 'balanced'")
            return StudentSizingHeuristics._balanced_sizing(teacher_config, compression_ratio)
    
    @staticmethod
    def _conservative_sizing(teacher_config: Dict[str, Any], ratio: float) -> Dict[str, Any]:
        """
        Conservative sizing: Primarily reduce depth, keep width mostly intact.
        Good for maintaining accuracy with moderate compression.
        """
        student = teacher_config.copy()
        
        # Reduce layers more aggressively
        depth_ratio = math.sqrt(ratio)  # Square root for smoother reduction
        student['num_layers'] = max(4, int(teacher_config['num_layers'] * depth_ratio))
        
        # Keep width mostly the same (slight reduction)
        width_ratio = 0.9 if ratio < 0.7 else 1.0
        student['hidden_size'] = _round_to_multiple(
            int(teacher_config['hidden_size'] * width_ratio), 
            64
        )
        
        # Adjust attention heads proportionally to hidden size
        student['num_attention_heads'] = _calculate_attention_heads(
            student['hidden_size'],
            teacher_config['num_attention_heads']
        )
        
        # Scale intermediate size
        student['intermediate_size'] = student['hidden_size'] * 4
        
        # Keep vocab size
        student['vocab_size'] = teacher_config['vocab_size']
        
        # Estimate parameters
        student['total_params'] = _estimate_params(student)
        
        LOG.info(f"Conservative sizing: {teacher_config['num_layers']} → {student['num_layers']} layers")
        return student
    
    @staticmethod
    def _balanced_sizing(teacher_config: Dict[str, Any], ratio: float) -> Dict[str, Any]:
        """
        Balanced sizing: Reduce both depth and width proportionally.
        Good balance between compression and accuracy.
        """
        student = teacher_config.copy()
        
        # Reduce layers
        student['num_layers'] = max(4, int(teacher_config['num_layers'] * math.sqrt(ratio)))
        
        # Reduce hidden size
        width_ratio = math.pow(ratio, 0.4)  # Less aggressive than square root
        student['hidden_size'] = _round_to_multiple(
            int(teacher_config['hidden_size'] * width_ratio),
            64
        )
        
        # Adjust attention heads
        student['num_attention_heads'] = _calculate_attention_heads(
            student['hidden_size'],
            teacher_config['num_attention_heads']
        )
        
        # Scale intermediate size (FFN)
        student['intermediate_size'] = student['hidden_size'] * 4
        
        # Keep vocab size
        student['vocab_size'] = teacher_config['vocab_size']
        
        # Estimate parameters
        student['total_params'] = _estimate_params(student)
        
        LOG.info(f"Balanced sizing: {teacher_config['num_layers']} → {student['num_layers']} layers, "
                f"{teacher_config['hidden_size']} → {student['hidden_size']} hidden")
        return student
    
    @staticmethod
    def _aggressive_sizing(teacher_config: Dict[str, Any], ratio: float) -> Dict[str, Any]:
        """
        Aggressive sizing: Heavily reduce depth, width, and attention.
        Maximum compression, may sacrifice some accuracy.
        """
        student = teacher_config.copy()
        
        # Aggressively reduce layers
        student['num_layers'] = max(3, int(teacher_config['num_layers'] * ratio))
        
        # Aggressively reduce hidden size
        width_ratio = math.pow(ratio, 0.6)
        student['hidden_size'] = _round_to_multiple(
            int(teacher_config['hidden_size'] * width_ratio),
            64
        )
        
        # Reduce attention heads more
        student['num_attention_heads'] = max(
            4,
            _calculate_attention_heads(student['hidden_size'], teacher_config['num_attention_heads']) - 2
        )
        
        # Scale intermediate size
        student['intermediate_size'] = student['hidden_size'] * 4
        
        # Keep vocab size
        student['vocab_size'] = teacher_config['vocab_size']
        
        # Estimate parameters
        student['total_params'] = _estimate_params(student)
        
        LOG.info(f"Aggressive sizing: {teacher_config['num_layers']} → {student['num_layers']} layers, "
                f"{teacher_config['hidden_size']} → {student['hidden_size']} hidden")
        return student


# Helper functions

def _round_to_multiple(value: int, multiple: int) -> int:
    """Round value to nearest multiple."""
    return int(round(value / multiple) * multiple)


def _calculate_attention_heads(hidden_size: int, teacher_heads: int) -> int:
    """
    Calculate number of attention heads for given hidden size.
    Ensure hidden_size is divisible by num_heads.
    """
    # Common head dimensions: 64, 96, 128
    for head_dim in [64, 96, 128]:
        heads = hidden_size // head_dim
        if heads > 0 and hidden_size % heads == 0:
            # Prefer head count close to teacher's
            return min(heads, teacher_heads)
    
    # Fallback: ensure divisibility
    for heads in range(min(12, teacher_heads), 0, -1):
        if hidden_size % heads == 0:
            return heads
    
    return 8  # Safe default


def _estimate_params(config: Dict[str, Any]) -> int:
    """
    Estimate total parameters for transformer architecture.
    
    Rough formula:
    - Embeddings: vocab_size * hidden_size
    - Encoder layers: num_layers * (12 * hidden_size^2 + 13 * hidden_size)
    - Approximation for attention, FFN, and layer norm
    """
    vocab_size = config['vocab_size']
    hidden_size = config['hidden_size']
    num_layers = config['num_layers']
    intermediate_size = config.get('intermediate_size', hidden_size * 4)
    
    # Embeddings
    embedding_params = vocab_size * hidden_size
    
    # Per-layer parameters (simplified)
    # Self-attention: 4 * (hidden * hidden) for Q, K, V, O projections
    attention_params = 4 * (hidden_size * hidden_size)
    # FFN: 2 * (hidden * intermediate) for up and down projections
    ffn_params = 2 * (hidden_size * intermediate_size)
    # Layer norms: 2 * 2 * hidden (scale + bias for 2 layer norms)
    ln_params = 4 * hidden_size
    
    layer_params = attention_params + ffn_params + ln_params
    
    # Total
    total = embedding_params + (num_layers * layer_params)
    
    return int(total)


__all__ = ['StudentSizingHeuristics']
