"""
Preflight Analysis Module
==========================

Automated model, data, and resource analysis before distillation.

Main Components:
    - ModelInspector: Model architecture analysis and compatibility checking
    - DataInspector: Dataset validation and task detection
    - ResourceProbe: Hardware resource profiling
    - PreflightAnalyzer: Main orchestrator for all preflight checks

Quick Start:
    ```python
    from zynthe.core.preflight import run_preflight_check
    
    report = run_preflight_check(
        teacher_model=teacher,
        student_model=student,
        dataset=train_dataset,
        config=config,
        save_report=True
    )
    
    if report['can_proceed']:
        # Use optimized config
        optimized_config = report['optimized_config']
        # Start training...
    else:
        # Fix issues
        print("Blockers:", report['blockers'])
    ```

Advanced Usage:
    ```python
    from zynthe.core.preflight import PreflightAnalyzer
    
    analyzer = PreflightAnalyzer(
        teacher_model=teacher,
        student_model=student,
        dataset=dataset,
        config=config
    )
    
    # Run full analysis
    report = analyzer.run_preflight(verbose=True)
    
    # Save reports
    analyzer.save_report(format='json')
    analyzer.save_report(format='txt')
    
    # Update config with optimizations
    updated_config = analyzer.update_config(
        save_path='optimized_config.yaml'
    )
    ```
"""

from .model_inspector import ModelInspector
from .data_inspector import DataInspector
from .resource_probe import ResourceProbe
from .analyser import PreflightAnalyzer, run_preflight_check

__all__ = [
    'ModelInspector',
    'DataInspector',
    'ResourceProbe',
    'PreflightAnalyzer',
    'run_preflight_check'
]

__version__ = '1.0.0'
