"""
Intelligent Agents for Knowledge Distillation

This module contains AI agents that automate various aspects of the distillation pipeline.
"""

from .teacher_agent import TeacherModelAgent, TeacherRecommendation, quick_teacher_setup

__all__ = [
    "TeacherModelAgent",
    "TeacherRecommendation", 
    "quick_teacher_setup"
]
