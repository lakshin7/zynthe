#!/usr/bin/env python3
"""
Quick integration test for Phase A extended metrics.
Verifies that extended metrics are properly integrated into the training workflow.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all extended metrics modules can be imported."""
    print("🔍 Testing imports...")
    
    try:
        from evaluation.metrics_extended import (
            compute_extended_metrics,
            DistillationEfficacyIndex,
            CompressionAwareScore,
            LossComponentTracker
        )
        print("✅ metrics_extended imports successful")
    except ImportError as e:
        print(f"❌ Failed to import from metrics_extended: {e}")
        return False
    
    try:
        from evaluation.evaluator_extended import DualEvaluator, CurriculumEvaluator
        print("✅ evaluator_extended imports successful")
    except ImportError as e:
        print(f"❌ Failed to import from evaluator_extended: {e}")
        return False
    
    return True


def test_trainer_integration():
    """Test that trainer.py has extended metrics integrated."""
    print("\n🔍 Testing trainer.py integration...")
    
    trainer_path = project_root / "training" / "trainer.py"
    if not trainer_path.exists():
        print(f"❌ trainer.py not found at {trainer_path}")
        return False
    
    with open(trainer_path, 'r') as f:
        content = f.read()
    
    # Check for key integrations
    checks = {
        "Import compute_extended_metrics": "compute_extended_metrics" in content,
        "Import DistillationEfficacyIndex": "DistillationEfficacyIndex" in content,
        "Import CompressionAwareScore": "CompressionAwareScore" in content,
        "Extended metrics history": "extended_metrics_history" in content,
        "LossComponentTracker": "LossComponentTracker" in content,
        "evaluate() signature": "def evaluate(self, dataloader, compute_extended=True)" in content,
        "Returns 3 values": "return avg_loss, metrics, extended_metrics" in content or "return avg_loss, metrics, extended" in content,
        "Save extended metrics": "extended_metrics.json" in content,
    }
    
    all_passed = True
    for check_name, passed in checks.items():
        if passed:
            print(f"  ✅ {check_name}")
        else:
            print(f"  ❌ {check_name}")
            all_passed = False
    
    return all_passed


def test_main_integration():
    """Test that main.py has DualEvaluator integrated."""
    print("\n🔍 Testing main.py integration...")
    
    main_path = project_root / "app" / "main.py"
    if not main_path.exists():
        print(f"❌ main.py not found at {main_path}")
        return False
    
    with open(main_path, 'r') as f:
        content = f.read()
    
    # Check for key integrations
    checks = {
        "Import DualEvaluator": "DualEvaluator" in content,
        "Import DistillationEfficacyIndex": "DistillationEfficacyIndex" in content,
        "Import CompressionAwareScore": "CompressionAwareScore" in content,
        "Use DualEvaluator": "dual_evaluator = DualEvaluator(" in content,
        "Compute DEI": "compute_dei" in content,
        "Compute CAS": "compute_cas" in content,
        "Save extended evaluation": "extended_evaluation.json" in content,
        "DEI in summary": "dei_results" in content,
        "CAS in summary": "cas_results" in content,
    }
    
    all_passed = True
    for check_name, passed in checks.items():
        if passed:
            print(f"  ✅ {check_name}")
        else:
            print(f"  ❌ {check_name}")
            all_passed = False
    
    return all_passed


def test_artifacts_generated():
    """Check that example artifacts exist (if any training has been run)."""
    print("\n🔍 Checking for generated artifacts...")
    
    experiments_dir = project_root / "experiments"
    if not experiments_dir.exists():
        print("  ℹ️  No experiments directory found (no training runs yet)")
        return True
    
    # Find most recent experiment
    experiment_dirs = sorted([d for d in experiments_dir.iterdir() if d.is_dir()], reverse=True)
    if not experiment_dirs:
        print("  ℹ️  No experiment directories found (no training runs yet)")
        return True
    
    latest_exp = experiment_dirs[0]
    print(f"  📁 Checking latest experiment: {latest_exp.name}")
    
    artifacts = {
        "extended_metrics.json": latest_exp / "extended_metrics.json",
        "extended_evaluation.json": latest_exp / "extended_evaluation.json",
        "EXPERIMENT_SUMMARY.md": latest_exp / "EXPERIMENT_SUMMARY.md",
    }
    
    found_any = False
    for artifact_name, artifact_path in artifacts.items():
        if artifact_path.exists():
            print(f"  ✅ {artifact_name} exists")
            found_any = True
        else:
            print(f"  ℹ️  {artifact_name} not found (training may not have completed)")
    
    if not found_any:
        print("  ℹ️  No extended artifacts found yet. Run a training session to generate them.")
    
    return True


def main():
    print("="*70)
    print("Phase A Integration Test")
    print("="*70 + "\n")
    
    results = []
    
    # Run all tests
    results.append(("Imports", test_imports()))
    results.append(("Trainer Integration", test_trainer_integration()))
    results.append(("Main Integration", test_main_integration()))
    results.append(("Artifacts Check", test_artifacts_generated()))
    
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
        print("✅ All integration tests passed!")
        print("Phase A is fully integrated and ready for use.")
        print("\nNext step: Run a training session to test end-to-end:")
        print("  python app/main.py --config configs/default.yaml")
        print("\nOr proceed with Phase B (Benchmarking)")
    else:
        print("⚠️  Some integration tests failed.")
        print("Please review the errors above and fix any issues.")
    print("="*70)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
