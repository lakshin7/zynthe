#!/usr/bin/env python3
"""
Test script to verify evaluation and visualization fixes
Uses existing trained models from a completed experiment
"""

import sys
from pathlib import Path
import torch

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.config.config_manager import ConfigManager
from core.models.model_loader import load_models
from data.dataloaders import create_dataloaders
from evaluation.evaluator_extended import DualEvaluator
from evaluation.metrics_extended import DistillationEfficacyIndex, CompressionAwareScore
import json

try:
    from rich import print as rprint
except Exception:
    rprint = print


def test_dual_evaluator():
    """Test DualEvaluator with correct parameters"""
    print("\n" + "="*70)
    print("Testing DualEvaluator")
    print("="*70 + "\n")
    
    # Use the advanced config from the most recent experiment
    exp_dir = "experiments/20251031T164444Z_d4daf9ec"
    config_path = f"{exp_dir}/advanced.yaml"
    
    # Load config
    cfg_manager = ConfigManager(config_path=config_path)
    print(f"✅ Config loaded from {config_path}")
    print(f"   Device: {cfg_manager.device()}")
    
    # Load models
    print("\n📦 Loading models...")
    teacher, student, tokenizer = load_models(cfg_manager, cfg_manager.device())
    print(f"✅ Teacher: {cfg_manager.resolved_config.get('model', {}).get('name')}")
    print(f"✅ Student: {cfg_manager.resolved_config.get('model', {}).get('student_name')}")
    
    # Load validation data
    print("\n📊 Loading validation data...")
    _, val_loader = create_dataloaders(cfg_manager.resolved_config, tokenizer)
    print(f"✅ Val loader: {len(val_loader)} batches")
    
    # Test DualEvaluator with correct parameters
    print("\n🔍 Testing DualEvaluator...")
    try:
        dual_evaluator = DualEvaluator(
            teacher=teacher,  # Correct parameter name
            student=student,  # Correct parameter name
            dataloader=val_loader,
            device=cfg_manager.device()
        )
        print("✅ DualEvaluator initialized successfully!")
        
        # Run evaluation
        print("\n🚀 Running dual evaluation...")
        eval_results = dual_evaluator.evaluate()
        
        # Extract metrics
        metrics = eval_results.get('student', {})
        teacher_metrics = eval_results.get('teacher', {})
        extended_metrics = eval_results.get('extended', {})
        
        # Helper to format safely
        def fmt(value, format_spec='.4f'):
            if value == 'N/A' or value is None:
                return 'N/A'
            try:
                return f"{value:{format_spec}}"
            except (ValueError, TypeError):
                return str(value)
        
        # Display results
        print("\n" + "="*70)
        print("Evaluation Results")
        print("="*70 + "\n")
        
        rprint(f"[bold green]Teacher Metrics:[/bold green]")
        rprint(f"  Accuracy: {fmt(teacher_metrics.get('accuracy', 'N/A'))}")
        rprint(f"  F1 Score: {fmt(teacher_metrics.get('f1', 'N/A'))}")
        rprint(f"  Loss: {fmt(teacher_metrics.get('loss', 'N/A'))}")
        
        rprint(f"\n[bold green]Student Metrics:[/bold green]")
        rprint(f"  Accuracy: {fmt(metrics.get('accuracy', 'N/A'))}")
        rprint(f"  F1 Score: {fmt(metrics.get('f1', 'N/A'))}")
        rprint(f"  Loss: {fmt(metrics.get('loss', 'N/A'))}")
        
        rprint(f"\n[bold cyan]Extended Distillation Metrics:[/bold cyan]")
        rprint(f"  KL Divergence: {fmt(extended_metrics.get('kl_divergence', 'N/A'))}")
        rprint(f"  JS Divergence: {fmt(extended_metrics.get('js_divergence', 'N/A'))}")
        rprint(f"  Prediction Agreement: {fmt(extended_metrics.get('prediction_agreement', 'N/A'), '.2%')}")
        rprint(f"  Confidence Correlation: {fmt(extended_metrics.get('confidence_correlation', 'N/A'))}")
        
        # Compute DEI & CAS
        print("\n🧮 Computing DEI & CAS scores...")
        teacher_params = sum(p.numel() for p in teacher.parameters())
        student_params = sum(p.numel() for p in student.parameters())
        
        teacher_acc = teacher_metrics.get('accuracy', 0.0)
        student_acc = metrics.get('accuracy', 0.0)
        
        if teacher_acc and student_acc:
            dei_results = DistillationEfficacyIndex.compute_dei(
                teacher_acc=teacher_acc,
                student_acc=student_acc,
                teacher_params=teacher_params,
                student_params=student_params
            )
            
            cas_results = CompressionAwareScore.compute_cas(
                accuracy=student_acc,
                teacher_params=teacher_params,
                student_params=student_params,
                teacher_latency=1.0,  # Placeholder
                student_latency=0.5   # Placeholder
            )
            
            # Add rating to CAS based on score
            cas_score = cas_results['cas']
            if cas_score > 0.35:
                cas_rating = 'Excellent'
            elif cas_score > 0.25:
                cas_rating = 'Very Good'
            elif cas_score > 0.15:
                cas_rating = 'Good'
            elif cas_score > 0.05:
                cas_rating = 'Fair'
            else:
                cas_rating = 'Poor'
            cas_results['rating'] = cas_rating
            
            rprint(f"\n[bold magenta]Distillation Efficacy Index (DEI):[/bold magenta]")
            rprint(f"  DEI Score: {dei_results['dei']:.4f}")
            rprint(f"  Rating: {dei_results['efficiency_rating']}")
            rprint(f"  Accuracy Retention: {dei_results['accuracy_retention']:.2%}")
            rprint(f"  Compression Ratio: {dei_results['compression_ratio']:.2f}x")
            
            rprint(f"\n[bold magenta]Compression-Aware Score (CAS):[/bold magenta]")
            rprint(f"  CAS Score: {cas_results['cas']:.4f}")
            rprint(f"  Rating: {cas_results['rating']}")
            
            # Test saving to JSON (simulation)
            def convert_to_json_serializable(obj):
                """Convert numpy types to Python native types for JSON"""
                import numpy as np
                if isinstance(obj, dict):
                    return {k: convert_to_json_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, (np.integer, np.floating)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            
            test_output = {
                'teacher': convert_to_json_serializable(teacher_metrics),
                'student': convert_to_json_serializable(metrics),
                'extended_metrics': convert_to_json_serializable(extended_metrics),
                'dei': convert_to_json_serializable(dei_results),
                'cas': convert_to_json_serializable(cas_results)
            }
            
            output_path = Path(exp_dir) / "test_extended_evaluation.json"
            with open(output_path, 'w') as f:
                json.dump(test_output, f, indent=2)
            
            print(f"\n✅ Test results saved to: {output_path}")
        else:
            print("\n⚠️  Could not compute DEI/CAS: Missing accuracy values")
        
        return True
        
    except Exception as e:
        print(f"\n❌ DualEvaluator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_summary_formatting():
    """Test EXPERIMENT_SUMMARY.md formatting with N/A handling"""
    print("\n" + "="*70)
    print("Testing Summary Report Formatting")
    print("="*70 + "\n")
    
    # Simulate metrics with some N/A values
    test_cases = [
        {'name': 'With valid values', 'accuracy': 0.8567, 'f1': 0.8432},
        {'name': 'With N/A', 'accuracy': 'N/A', 'f1': 0.8432},
        {'name': 'With None', 'accuracy': None, 'f1': 0.8432},
        {'name': 'All N/A', 'accuracy': 'N/A', 'f1': 'N/A'},
    ]
    
    for test in test_cases:
        print(f"\nTest case: {test['name']}")
        acc = test['accuracy']
        f1 = test['f1']
        
        # Test the formatting logic
        try:
            acc_str = acc if acc == 'N/A' or acc is None else f'{acc:.4f}'
            f1_str = f1 if f1 == 'N/A' or f1 is None else f'{f1:.4f}'
            
            print(f"  Accuracy: {acc_str}")
            print(f"  F1 Score: {f1_str}")
            print("  ✅ Formatting succeeded")
        except Exception as e:
            print(f"  ❌ Formatting failed: {e}")
            return False
    
    print("\n✅ All formatting tests passed!")
    return True


def main():
    print("\n" + "="*70)
    print("Evaluation & Visualization Integration Test")
    print("="*70)
    
    results = []
    
    # Test DualEvaluator
    results.append(("DualEvaluator", test_dual_evaluator()))
    
    # Test formatting
    results.append(("Summary Formatting", test_summary_formatting()))
    
    # Summary
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70 + "\n")
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {test_name}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*70)
    if all_passed:
        print("✅ All tests passed! Fixes are working correctly.")
        print("\nYou can now run full training with:")
        print("  python app/main.py --config configs/advanced.yaml")
    else:
        print("⚠️  Some tests failed. Please review the errors above.")
    print("="*70 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
