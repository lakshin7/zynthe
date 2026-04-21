#!/usr/bin/env python3
"""
Check and summarize generated visualization artifacts from distillation experiment.
"""
import sys
from pathlib import Path

def check_artifacts(experiment_dir):
    """
    Verify presence of expected visualization artifacts in experiment directory.
    
    Expected artifacts:
    - Student micro-series: student_epoch{N}_train_micro.png, student_epoch{N}_eval_micro.png
    - Teacher micro-series: teacher_epoch{N}_train_micro.png, teacher_epoch{N}_eval_micro.png
    - Training curves: training_curves.png (student), teacher_training_curves.png
    - Confusion matrices: student_confusion/confusion_matrix.png, teacher_confusion/confusion_matrix.png
    - Metrics: extended_metrics.json, extended_evaluation.json, EXPERIMENT_SUMMARY.md
    """
    exp_path = Path(experiment_dir)
    
    if not exp_path.exists():
        print(f"❌ Experiment directory not found: {experiment_dir}")
        return False
    
    print(f"\n{'='*70}")
    print(f"📂 Checking artifacts in: {exp_path.name}")
    print(f"{'='*70}\n")
    
    # Track findings
    found = {
        'student_train_micro': [],
        'student_eval_micro': [],
        'teacher_train_micro': [],
        'teacher_eval_micro': [],
        'training_curves': [],
        'teacher_curves': [],
        'student_confusion': [],
        'teacher_confusion': [],
        'metrics': [],
    }
    
    # Student micro-series
    found['student_train_micro'] = sorted(exp_path.glob('student_epoch*_micro.png'))
    found['student_eval_micro'] = sorted(exp_path.glob('student_epoch*_eval_micro.png'))
    
    # Teacher micro-series
    found['teacher_train_micro'] = sorted(exp_path.glob('teacher_epoch*_train_micro.png'))
    found['teacher_eval_micro'] = sorted(exp_path.glob('teacher_epoch*_eval_micro.png'))
    
    # Training curves
    if (exp_path / 'training_curves.png').exists():
        found['training_curves'].append(exp_path / 'training_curves.png')
    if (exp_path / 'teacher_training_curves.png').exists():
        found['teacher_curves'].append(exp_path / 'teacher_training_curves.png')
    
    # Confusion matrices
    student_cm = exp_path / 'student_confusion' / 'confusion_matrix.png'
    teacher_cm = exp_path / 'teacher_confusion' / 'confusion_matrix.png'
    if student_cm.exists():
        found['student_confusion'].append(student_cm)
    if teacher_cm.exists():
        found['teacher_confusion'].append(teacher_cm)
    
    # Metrics
    for metric_file in ['extended_metrics.json', 'extended_evaluation.json', 'EXPERIMENT_SUMMARY.md', 'metrics.json']:
        if (exp_path / metric_file).exists():
            found['metrics'].append(exp_path / metric_file)
    
    # Print findings
    all_ok = True
    
    print("🎨 Student Visualizations:")
    if found['student_train_micro']:
        print(f"  ✅ Train micro-series: {len(found['student_train_micro'])} files")
        for f in found['student_train_micro']:
            print(f"     - {f.name}")
    else:
        print("  ❌ No student train micro-series found")
        all_ok = False
    
    if found['student_eval_micro']:
        print(f"  ✅ Eval micro-series: {len(found['student_eval_micro'])} files")
        for f in found['student_eval_micro']:
            print(f"     - {f.name}")
    else:
        print("  ❌ No student eval micro-series found")
        all_ok = False
    
    if found['training_curves']:
        print(f"  ✅ Training curves: {found['training_curves'][0].name}")
    else:
        print("  ❌ No student training_curves.png found")
        all_ok = False
    
    if found['student_confusion']:
        print("  ✅ Confusion matrix: student_confusion/confusion_matrix.png")
    else:
        print("  ⚠️  No student confusion matrix found")
    
    print("\n👨‍🏫 Teacher Visualizations:")
    if found['teacher_train_micro']:
        print(f"  ✅ Train micro-series: {len(found['teacher_train_micro'])} files")
        for f in found['teacher_train_micro']:
            print(f"     - {f.name}")
    else:
        print("  ⚠️  No teacher train micro-series found (check if teacher training was enabled)")
    
    if found['teacher_eval_micro']:
        print(f"  ✅ Eval micro-series: {len(found['teacher_eval_micro'])} files")
        for f in found['teacher_eval_micro']:
            print(f"     - {f.name}")
    else:
        print("  ⚠️  No teacher eval micro-series found")
    
    if found['teacher_curves']:
        print(f"  ✅ Training curves: {found['teacher_curves'][0].name}")
    else:
        print("  ⚠️  No teacher_training_curves.png found")
    
    if found['teacher_confusion']:
        print("  ✅ Confusion matrix: teacher_confusion/confusion_matrix.png")
    else:
        print("  ⚠️  No teacher confusion matrix found")
    
    print("\n📊 Metrics & Reports:")
    if found['metrics']:
        print(f"  ✅ {len(found['metrics'])} metric/report files:")
        for f in found['metrics']:
            print(f"     - {f.name}")
    else:
        print("  ❌ No metrics files found")
        all_ok = False
    
    print(f"\n{'='*70}")
    if all_ok:
        print("✅ All essential artifacts present!")
    else:
        print("⚠️  Some artifacts missing — check training logs")
    print(f"{'='*70}\n")
    
    return all_ok


def main():
    if len(sys.argv) > 1:
        experiment_dir = sys.argv[1]
    else:
        # Find the latest experiment directory
        experiments = sorted(Path('experiments').glob('*'), key=lambda x: x.stat().st_mtime, reverse=True)
        if not experiments:
            print("❌ No experiments found in experiments/ directory")
            return 1
        experiment_dir = str(experiments[0])
    
    success = check_artifacts(experiment_dir)
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
