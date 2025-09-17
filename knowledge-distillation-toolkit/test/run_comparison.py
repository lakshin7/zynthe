#!/usr/bin/env python3
"""
Quick script to run model comparison on the latest experiment.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

def find_latest_experiment():
    """Find the most recent experiment directory."""
    experiments_dir = Path("experiments")
    if not experiments_dir.exists():
        print("❌ No experiments directory found!")
        return None
    
    # Get all experiment directories
    exp_dirs = [d for d in experiments_dir.iterdir() if d.is_dir()]
    if not exp_dirs:
        print("❌ No experiment directories found!")
        return None
    
    # Sort by modification time (newest first)
    latest_exp = max(exp_dirs, key=lambda x: x.stat().st_mtime)
    return latest_exp

def main():
    """Run comparison on latest experiment."""
    print("🔍 Finding latest experiment...")
    
    latest_exp = find_latest_experiment()
    if not latest_exp:
        return
    
    print(f"📁 Latest experiment: {latest_exp}")
    
    # Check if student model exists
    student_model_path = latest_exp / "student_model"
    if not student_model_path.exists():
        print(f"❌ Student model not found at {student_model_path}")
        return
    
    print("✅ Student model found!")
    
    # Run comparison
    from compare_models import ModelComparator
    
    comparator = ModelComparator()
    teacher_results, student_results = comparator.run_comparison(str(latest_exp))
    
    print("\n🎉 Comparison completed!")

if __name__ == "__main__":
    main()
