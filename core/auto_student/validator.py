"""
Student Architecture Validator
==============================

Validates generated student architectures for correctness and feasibility.

Checks:
- Dimension divisibility (hidden_size % num_heads == 0)
- Reasonable parameter counts
- Memory feasibility
- Architecture consistency
"""

import logging
from typing import Dict, Any, List, Tuple

LOG = logging.getLogger(__name__)


class StudentValidator:
    """
    Validates student architecture configurations.
    """
    
    # Constraints for reasonable architectures
    MIN_LAYERS = 2
    MAX_LAYERS = 48
    MIN_HIDDEN_SIZE = 128
    MAX_HIDDEN_SIZE = 4096
    MIN_ATTENTION_HEADS = 2
    MAX_ATTENTION_HEADS = 32
    MIN_PARAMS = 1_000_000  # 1M
    MAX_PARAMS = 1_000_000_000  # 1B (for reasonable training on Mac M2)
    
    @staticmethod
    def validate(student_config: Dict[str, Any], strict: bool = False) -> Tuple[bool, List[str]]:
        """
        Validate student architecture configuration.
        
        Args:
            student_config: Student architecture config
            strict: If True, treat warnings as errors
            
        Returns:
            (is_valid, list_of_issues)
        """
        issues = []
        
        # Check required fields
        required_fields = ['num_layers', 'hidden_size', 'num_attention_heads']
        for field in required_fields:
            if field not in student_config:
                issues.append(f"Missing required field: {field}")
        
        if issues:
            return False, issues
        
        # Extract values
        num_layers = student_config['num_layers']
        hidden_size = student_config['hidden_size']
        num_heads = student_config['num_attention_heads']
        total_params = student_config.get('total_params', 0)
        
        # Validation checks
        
        # 1. Layer count
        if num_layers < StudentValidator.MIN_LAYERS:
            issues.append(f"Too few layers: {num_layers} < {StudentValidator.MIN_LAYERS}")
        elif num_layers > StudentValidator.MAX_LAYERS:
            issues.append(f"Too many layers: {num_layers} > {StudentValidator.MAX_LAYERS}")
        
        # 2. Hidden size
        if hidden_size < StudentValidator.MIN_HIDDEN_SIZE:
            issues.append(f"Hidden size too small: {hidden_size} < {StudentValidator.MIN_HIDDEN_SIZE}")
        elif hidden_size > StudentValidator.MAX_HIDDEN_SIZE:
            issues.append(f"Hidden size too large: {hidden_size} > {StudentValidator.MAX_HIDDEN_SIZE}")
        
        # 3. Attention heads
        if num_heads < StudentValidator.MIN_ATTENTION_HEADS:
            issues.append(f"Too few attention heads: {num_heads} < {StudentValidator.MIN_ATTENTION_HEADS}")
        elif num_heads > StudentValidator.MAX_ATTENTION_HEADS:
            issues.append(f"Too many attention heads: {num_heads} > {StudentValidator.MAX_ATTENTION_HEADS}")
        
        # 4. Divisibility: hidden_size must be divisible by num_heads
        if hidden_size % num_heads != 0:
            issues.append(f"Hidden size ({hidden_size}) not divisible by num_heads ({num_heads})")
        
        # 5. Parameter count
        if total_params > 0:
            if total_params < StudentValidator.MIN_PARAMS:
                issues.append(f"Too few parameters: {total_params:,} < {StudentValidator.MIN_PARAMS:,}")
            elif total_params > StudentValidator.MAX_PARAMS:
                issues.append(f"Too many parameters: {total_params:,} > {StudentValidator.MAX_PARAMS:,}")
        
        # 6. Intermediate size check (FFN)
        if 'intermediate_size' in student_config:
            intermediate_size = student_config['intermediate_size']
            # Typically intermediate_size = 4 * hidden_size
            expected_ratio = intermediate_size / hidden_size
            if expected_ratio < 2 or expected_ratio > 8:
                msg = f"Unusual intermediate_size ratio: {expected_ratio:.1f}x (expected 3-5x)"
                if strict:
                    issues.append(msg)
                else:
                    LOG.warning(msg)
        
        # 7. Head dimension check
        head_dim = hidden_size // num_heads
        if head_dim < 32:
            msg = f"Small head dimension: {head_dim} (may hurt performance)"
            if strict:
                issues.append(msg)
            else:
                LOG.warning(msg)
        elif head_dim > 256:
            msg = f"Large head dimension: {head_dim} (may be inefficient)"
            if strict:
                issues.append(msg)
            else:
                LOG.warning(msg)
        
        # Determine validity
        is_valid = len(issues) == 0
        
        if is_valid:
            LOG.info(f"✓ Validation passed: {num_layers} layers, {hidden_size} hidden, "
                    f"{num_heads} heads, {total_params:,} params")
        else:
            LOG.error(f"✗ Validation failed with {len(issues)} issue(s)")
            for issue in issues:
                LOG.error(f"  - {issue}")
        
        return is_valid, issues
    
    @staticmethod
    def check_memory_feasibility(
        student_config: Dict[str, Any],
        batch_size: int = 8,
        seq_length: int = 128,
        available_memory_gb: float = 8.0
    ) -> Tuple[bool, float]:
        """
        Estimate memory usage and check feasibility.
        
        Args:
            student_config: Student architecture
            batch_size: Training batch size
            seq_length: Sequence length
            available_memory_gb: Available memory in GB
            
        Returns:
            (is_feasible, estimated_memory_gb)
        """
        # Rough memory estimation
        # Model parameters
        params = student_config.get('total_params', 0)
        
        # Memory breakdown:
        # 1. Model weights: params * 4 bytes (fp32)
        model_memory = params * 4 / (1024**3)  # GB
        
        # 2. Optimizer states (AdamW): 2x model weights (momentum + variance)
        optimizer_memory = model_memory * 2
        
        # 3. Activations: batch_size * seq_length * hidden_size * num_layers * 4
        hidden_size = student_config['hidden_size']
        num_layers = student_config['num_layers']
        activation_memory = (batch_size * seq_length * hidden_size * num_layers * 4) / (1024**3)
        
        # 4. Gradients: same as model weights
        gradient_memory = model_memory
        
        # 5. Overhead: 20% buffer for misc allocations
        overhead = 0.2 * (model_memory + optimizer_memory + activation_memory + gradient_memory)
        
        # Total
        total_memory = model_memory + optimizer_memory + activation_memory + gradient_memory + overhead
        
        is_feasible = total_memory <= available_memory_gb
        
        if is_feasible:
            LOG.info(f"✓ Memory feasible: {total_memory:.2f} GB <= {available_memory_gb:.2f} GB")
        else:
            LOG.warning(f"⚠ Memory may be tight: {total_memory:.2f} GB > {available_memory_gb:.2f} GB")
            LOG.warning("  Consider reducing batch_size or seq_length")
        
        return is_feasible, total_memory
    
    @staticmethod
    def suggest_fixes(student_config: Dict[str, Any], issues: List[str]) -> Dict[str, Any]:
        """
        Suggest fixes for validation issues.
        
        Args:
            student_config: Student config with issues
            issues: List of validation issues
            
        Returns:
            Fixed student config (best effort)
        """
        fixed_config = student_config.copy()
        
        for issue in issues:
            if "not divisible by num_heads" in issue:
                # Fix divisibility issue
                hidden_size = fixed_config['hidden_size']
                num_heads = fixed_config['num_attention_heads']
                
                # Round hidden_size to be divisible by num_heads
                fixed_hidden = (hidden_size // num_heads) * num_heads
                if fixed_hidden < StudentValidator.MIN_HIDDEN_SIZE:
                    fixed_hidden = num_heads * (StudentValidator.MIN_HIDDEN_SIZE // num_heads + 1)
                
                LOG.info(f"Fixed hidden_size: {hidden_size} → {fixed_hidden}")
                fixed_config['hidden_size'] = fixed_hidden
                fixed_config['intermediate_size'] = fixed_hidden * 4
            
            elif "Too few layers" in issue:
                fixed_config['num_layers'] = StudentValidator.MIN_LAYERS
                LOG.info(f"Fixed num_layers → {StudentValidator.MIN_LAYERS}")
            
            elif "Too few attention heads" in issue:
                fixed_config['num_attention_heads'] = StudentValidator.MIN_ATTENTION_HEADS
                LOG.info(f"Fixed num_attention_heads → {StudentValidator.MIN_ATTENTION_HEADS}")
        
        return fixed_config


__all__ = ['StudentValidator']
