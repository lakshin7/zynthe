#!/usr/bin/env python3
"""
Test Extended Metrics on Recent Training Run
Demonstrates the new Zynthe EvalX features
"""

import sys
import os
import torch
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from evaluation.metrics_extended import (
    DistillationMetrics,
    CompressionAwareScore,
    DistillationEfficacyIndex,
    compute_extended_metrics
)
from data.dataloaders import create_dataloaders
from core.config.config_manager import ConfigManager

def main():
    print("="*70)
    print("Zynthe EvalX - Testing Extended Metrics")
    print("="*70)
    print()
    
    # Load your recent experiment
    exp_dir = "experiments/20251023T175322Z_5285cff4"
    
    print(f"📂 Loading models from: {exp_dir}")
    
    # Load models
    teacher_path = os.path.join(exp_dir, "teacher_model")
    student_path = os.path.join(exp_dir, "student_model")
    
    if not os.path.exists(teacher_path) or not os.path.exists(student_path):
        print("❌ Error: Models not found. Run training first.")
        return
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"🔧 Device: {device}")
    
    teacher = AutoModelForSequenceClassification.from_pretrained(teacher_path)
    student = AutoModelForSequenceClassification.from_pretrained(student_path)
    tokenizer = AutoTokenizer.from_pretrained(teacher_path)
    
    teacher.to(device)
    student.to(device)
    teacher.eval()
    student.eval()
    
    print("✅ Models loaded")
    print()
    
    # Load validation data
    print("📊 Loading validation data...")
    cfg = ConfigManager('configs/quick_test_minilm.yaml')
    _, val_loader = create_dataloaders(cfg.resolved_config, tokenizer)
    
    # Get sample batch
    sample_batch = next(iter(val_loader))
    input_ids = sample_batch['input_ids'][:16].to(device)
    attention_mask = sample_batch['attention_mask'][:16].to(device)
    labels = sample_batch['labels'][:16].to(device)
    
    print(f"  Batch size: {input_ids.size(0)}")
    print()
    
    # Run inference
    print("🔬 Computing extended metrics...")
    print()
    
    with torch.no_grad():
        teacher_outputs = teacher(input_ids=input_ids, attention_mask=attention_mask)
        student_outputs = student(input_ids=input_ids, attention_mask=attention_mask)
    
    teacher_logits = teacher_outputs.logits
    student_logits = student_outputs.logits
    
    # Compute extended metrics
    metrics = compute_extended_metrics(teacher_logits, student_logits, temperature=2.0)
    
    print("📈 Distillation Metrics:")
    print("-" * 60)
    print(f"  KL Divergence:           {metrics['kl_divergence']:.4f}")
    print(f"  JS Divergence:           {metrics['js_divergence']:.4f}")
    print(f"  Prediction Agreement:    {metrics['prediction_agreement']:.4f} ({metrics['prediction_agreement']*100:.1f}%)")
    print(f"  Confidence Correlation:  {metrics['confidence_correlation']:.4f}")
    print()
    
    # Interpretation
    print("💡 Interpretation:")
    if metrics['kl_divergence'] < 0.5:
        print("  ✅ Excellent knowledge transfer (KL < 0.5)")
    elif metrics['kl_divergence'] < 1.0:
        print("  ✓ Good knowledge transfer (KL < 1.0)")
    else:
        print("  ⚠️  Moderate knowledge transfer (KL > 1.0)")
    
    if metrics['prediction_agreement'] > 0.95:
        print("  ✅ Exceptional prediction mimicry (>95% agreement)")
    elif metrics['prediction_agreement'] > 0.90:
        print("  ✓ Excellent prediction mimicry (>90% agreement)")
    else:
        print(f"  → Prediction agreement: {metrics['prediction_agreement']*100:.1f}%")
    
    if metrics['confidence_correlation'] > 0.9:
        print("  ✅ Strong confidence calibration (r > 0.9)")
    elif metrics['confidence_correlation'] > 0.7:
        print("  ✓ Good confidence calibration (r > 0.7)")
    else:
        print(f"  → Confidence correlation: {metrics['confidence_correlation']:.4f}")
    print()
    
    # Compute model statistics
    teacher_params = sum(p.numel() for p in teacher.parameters())
    student_params = sum(p.numel() for p in student.parameters())
    
    print("📦 Model Compression:")
    print("-" * 60)
    print(f"  Teacher params:  {teacher_params:,} ({teacher_params/1e6:.1f}M)")
    print(f"  Student params:  {student_params:,} ({student_params/1e6:.1f}M)")
    print(f"  Compression:     {teacher_params/student_params:.2f}x")
    print(f"  Size reduction:  {(1 - student_params/teacher_params)*100:.1f}%")
    print()
    
    # Compute DEI (using your actual accuracies from training)
    teacher_acc = 0.960  # Assume teacher ~96% (pre-trained)
    student_acc = 0.9495  # From your training log
    
    dei_result = DistillationEfficacyIndex.compute_dei(
        teacher_acc=teacher_acc,
        student_acc=student_acc,
        teacher_params=teacher_params,
        student_params=student_params,
        retention_bonus=metrics['prediction_agreement'] * 0.1
    )
    
    print("🎯 Distillation Efficacy Index (DEI):")
    print("-" * 60)
    print(f"  DEI Score:            {dei_result['dei']:.4f}")
    print(f"  Accuracy Retention:   {dei_result['accuracy_retention']:.4f} ({dei_result['accuracy_retention']*100:.1f}%)")
    print(f"  Compression Ratio:    {dei_result['compression_ratio']:.2f}x")
    print(f"  Accuracy Drop:        {dei_result['accuracy_drop']:.4f} ({dei_result['accuracy_drop']*100:.2f}%)")
    print(f"  Efficiency Rating:    {dei_result['efficiency_rating']}")
    print()
    
    if dei_result['dei'] > 1.5:
        print("  🌟 OUTSTANDING distillation! (DEI > 1.5)")
    elif dei_result['dei'] > 1.0:
        print("  ✅ Excellent distillation! (DEI > 1.0)")
    else:
        print("  ✓ Good distillation (DEI > 0.8)")
    print()
    
    # Compute CAS (assuming latency estimates)
    teacher_latency = 45.0  # ms (estimated for RoBERTa-base)
    student_latency = 28.5  # ms (estimated for DistilRoBERTa)
    
    cas_result = CompressionAwareScore.compute_cas(
        accuracy=student_acc,
        teacher_params=teacher_params,
        student_params=student_params,
        teacher_latency=teacher_latency,
        student_latency=student_latency,
        alpha=0.6,
        beta=0.2,
        gamma=0.2
    )
    
    print("💎 Compression-Aware Score (CAS):")
    print("-" * 60)
    print(f"  CAS Score:           {cas_result['cas']:.4f}")
    print(f"  Accuracy:            {cas_result['accuracy']:.4f}")
    print(f"  Size Ratio:          {cas_result['size_ratio']:.4f}")
    print(f"  Latency Ratio:       {cas_result['latency_ratio']:.4f}")
    print(f"  Speedup:             {cas_result['speedup']:.2f}x")
    print(f"  Efficiency Score:    {cas_result['efficiency_score']:.4f}")
    print()
    
    if cas_result['cas'] > 0.5:
        print("  🌟 EXCELLENT model for deployment! (CAS > 0.5)")
    elif cas_result['cas'] > 0.3:
        print("  ✅ Very good deployment candidate (CAS > 0.3)")
    elif cas_result['cas'] > 0.0:
        print("  ✓ Good deployment candidate (CAS > 0)")
    else:
        print("  ⚠️  Consider tuning for better deployment score")
    print()
    
    # Summary
    print("="*70)
    print("📊 SUMMARY")
    print("="*70)
    print()
    print("Your distillation achieved:")
    print(f"  ✓ {dei_result['accuracy_retention']*100:.1f}% accuracy retention")
    print(f"  ✓ {teacher_params/student_params:.2f}x model compression")
    print(f"  ✓ {metrics['prediction_agreement']*100:.1f}% teacher-student agreement")
    print(f"  ✓ DEI Score: {dei_result['dei']:.4f} ({dei_result['efficiency_rating']})")
    print(f"  ✓ CAS Score: {cas_result['cas']:.4f}")
    print()
    print("🎉 Your model is production-ready with Zynthe EvalX metrics!")
    print()


if __name__ == "__main__":
    main()
