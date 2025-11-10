"""
Auto-Student Module
===================

Automatically generates optimal student architectures from teacher models.

Features:
- Heuristic-based student sizing
- Architecture validation
- Template-based generation
- Integration with existing distillation pipeline

Usage:
    from core.auto_student import AutoStudentBuilder
    
    builder = AutoStudentBuilder(teacher_name="bert-base-uncased")
    student_config = builder.generate(compression_ratio=0.5)
"""

from .auto_student_builder import AutoStudentBuilder
from .heuristics import StudentSizingHeuristics
from .validator import StudentValidator

__all__ = [
    'AutoStudentBuilder',
    'StudentSizingHeuristics',
    'StudentValidator',
]

__version__ = '1.0.0'
