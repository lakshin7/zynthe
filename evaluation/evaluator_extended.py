"""
Zynthe EvalX - Enhanced Evaluator with Dual-Model Comparison
Supports side-by-side teacher-student evaluation with extended metrics
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Any, Callable
import numpy as np
import time
from evaluation.metrics import compute_all_metrics
from evaluation.metrics_extended import (
    compute_extended_metrics,
    DistillationEfficacyIndex,
    PerformanceProfiler,
    CompressionAwareScore
)
from evaluation.evaluation_report import EvaluationReport


class DualEvaluator:
    """
    Evaluate teacher and student models side-by-side for real-time comparison.
    Doubles evaluation efficiency and provides distillation-specific metrics.
    """
    
    def __init__(self,
                 teacher: nn.Module,
                 student: nn.Module,
                 dataloader,
                 device: str,
                 modality: str = "text",
                 temperature: float = 2.0,
                 compute_features: bool = False,
                 progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
                 update_frequency: int = 10):
        """
        Args:
            teacher: Teacher model
            student: Student model
            dataloader: Validation/test dataloader
            device: Device (cpu/cuda/mps)
            modality: Data modality (text/vision/multimodal)
            temperature: Temperature for KL divergence
            compute_features: Whether to extract and compare features
            progress_callback: Optional callback for real-time progress updates
            update_frequency: Update progress every N batches
        """
        self.teacher = teacher
        self.student = student
        self.dataloader = dataloader
        self.device = device
        self.modality = modality
        self.temperature = temperature
        self.compute_features = compute_features
        
        # Real-time progress streaming for UI
        self.progress_callback = progress_callback
        self.update_frequency = update_frequency
        
        # Move models to device
        self.teacher.to(device)
        self.student.to(device)
        
        # Set to eval mode
        self.teacher.eval()
        self.student.eval()
    
    def evaluate(self, profile_performance: bool = False) -> EvaluationReport:
        """
        Run dual evaluation with extended metrics.
        
        Args:
            profile_performance: Whether to profile inference latency
            
        Returns:
            Comprehensive evaluation results
        """
        print("[DUAL EVAL] Starting side-by-side teacher-student evaluation...")
        
        teacher_preds = []
        student_preds = []
        all_labels = []
        
        teacher_logits_all = []
        student_logits_all = []
        
        teacher_correct = 0
        student_correct = 0
        agreement_count = 0
        total_samples = 0
        
        kl_divs = []
        js_divs = []
        confidence_corrs = []
        
        # Timing
        teacher_time = 0.0
        student_time = 0.0
        
        # Calculate total batches for progress tracking
        total_batches = len(self.dataloader) if hasattr(self.dataloader, '__len__') else None
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.dataloader):
                # Extract batch data
                if isinstance(batch, dict):
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch.get('attention_mask')
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(self.device)
                    labels = batch['labels'].to(self.device)
                else:
                    input_ids, labels = batch[:2]
                    input_ids = input_ids.to(self.device)
                    labels = labels.to(self.device)
                    attention_mask = batch[2].to(self.device) if len(batch) > 2 else None
                
                batch_size = input_ids.size(0)
                
                # Teacher inference
                start_t = time.perf_counter()
                teacher_outputs = self.teacher(input_ids=input_ids, attention_mask=attention_mask)
                teacher_time += time.perf_counter() - start_t
                
                # Student inference
                start_s = time.perf_counter()
                student_outputs = self.student(input_ids=input_ids, attention_mask=attention_mask)
                student_time += time.perf_counter() - start_s
                
                # Extract logits
                teacher_logits = teacher_outputs.logits if hasattr(teacher_outputs, 'logits') else teacher_outputs[0]
                student_logits = student_outputs.logits if hasattr(student_outputs, 'logits') else student_outputs[0]
                
                # Predictions
                teacher_pred = torch.argmax(teacher_logits, dim=-1)
                student_pred = torch.argmax(student_logits, dim=-1)
                
                # Accuracy tracking
                teacher_correct += (teacher_pred == labels).sum().item()
                student_correct += (student_pred == labels).sum().item()
                agreement_count += (teacher_pred == student_pred).sum().item()
                total_samples += batch_size
                
                # Store for metrics
                teacher_preds.extend(teacher_pred.cpu().numpy().tolist())
                student_preds.extend(student_pred.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())
                
                teacher_logits_all.append(teacher_logits.cpu())
                student_logits_all.append(student_logits.cpu())
                
                # Compute batch-wise extended metrics
                batch_metrics = compute_extended_metrics(
                    teacher_logits, student_logits, 
                    temperature=self.temperature
                )
                kl_divs.append(batch_metrics['kl_divergence'])
                js_divs.append(batch_metrics['js_divergence'])
                confidence_corrs.append(batch_metrics['confidence_correlation'])
                
                # ========== REAL-TIME PROGRESS STREAMING ==========
                # Send progress updates to UI via WebSocket
                if self.progress_callback and (batch_idx + 1) % self.update_frequency == 0:
                    running_teacher_acc = teacher_correct / total_samples if total_samples > 0 else 0
                    running_student_acc = student_correct / total_samples if total_samples > 0 else 0
                    running_agreement = agreement_count / total_samples if total_samples > 0 else 0
                    
                    progress_payload = {
                        'type': 'evaluation_progress',
                        'stage': 'dual_evaluation',
                        'batch': batch_idx + 1,
                        'total_batches': total_batches,
                        'progress': ((batch_idx + 1) / total_batches * 100) if total_batches else None,
                        'samples_processed': total_samples,
                        'teacher_accuracy': running_teacher_acc,
                        'student_accuracy': running_student_acc,
                        'prediction_agreement': running_agreement,
                        'avg_kl_divergence': np.mean(kl_divs) if kl_divs else None,
                        'teacher_latency_ms': (teacher_time / total_samples * 1000) if total_samples > 0 else None,
                        'student_latency_ms': (student_time / total_samples * 1000) if total_samples > 0 else None
                    }
                    
                    try:
                        self.progress_callback(progress_payload)
                    except Exception as e:
                        print(f"[WARNING] Progress callback failed: {e}")
                # ================================================
                
                if (batch_idx + 1) % 20 == 0:
                    print(f"[DUAL EVAL] Processed {batch_idx + 1}/{len(self.dataloader)} batches")
        
        # Compute standard metrics
        teacher_metrics = compute_all_metrics(teacher_preds, all_labels)
        student_metrics = compute_all_metrics(student_preds, all_labels)
        
        # Aggregate extended metrics
        teacher_acc = teacher_correct / total_samples
        student_acc = student_correct / total_samples
        prediction_agreement = agreement_count / total_samples
        
        # Concatenate logits for global metrics
        teacher_logits_cat = torch.cat(teacher_logits_all, dim=0)
        student_logits_cat = torch.cat(student_logits_all, dim=0)
        
        global_metrics = compute_extended_metrics(
            teacher_logits_cat, student_logits_cat,
            temperature=self.temperature
        )
        
        # Compute DEI
        teacher_params = sum(p.numel() for p in self.teacher.parameters())
        student_params = sum(p.numel() for p in self.student.parameters())
        
        dei_result = DistillationEfficacyIndex.compute_dei(
            teacher_acc=teacher_acc,
            student_acc=student_acc,
            teacher_params=teacher_params,
            student_params=student_params,
            retention_bonus=prediction_agreement * 0.1  # Bonus for high agreement
        )
        
        # Performance profiling
        perf_comparison = {}
        if profile_performance:
            # Profile with sample batch
            sample_batch = next(iter(self.dataloader))
            sample_input = sample_batch['input_ids'][:8].to(self.device)
            sample_mask = sample_batch['attention_mask'][:8].to(self.device) if 'attention_mask' in sample_batch else None
            
            teacher_profile = PerformanceProfiler.profile_inference(
                self.teacher, sample_input, sample_mask, str(self.device)  # type: ignore[arg-type]
            )
            student_profile = PerformanceProfiler.profile_inference(
                self.student, sample_input, sample_mask, str(self.device)  # type: ignore[arg-type]
            )
            perf_comparison = PerformanceProfiler.compare_models(teacher_profile, student_profile)
            
            # Compute CAS
            cas_result = CompressionAwareScore.compute_cas(
                accuracy=student_acc,
                teacher_params=teacher_params,
                student_params=student_params,
                teacher_latency=teacher_profile['mean_latency_ms'],
                student_latency=student_profile['mean_latency_ms']
            )
        else:
            # Use averaged timing from evaluation
            avg_teacher_time = (teacher_time / total_samples) * 1000  # ms
            avg_student_time = (student_time / total_samples) * 1000  # ms
            
            cas_result = CompressionAwareScore.compute_cas(
                accuracy=student_acc,
                teacher_params=teacher_params,
                student_params=student_params,
                teacher_latency=avg_teacher_time,
                student_latency=avg_student_time
            )
            
            perf_comparison = {
                'speedup': avg_teacher_time / avg_student_time if avg_student_time > 0 else 0,
                'teacher_latency_ms': avg_teacher_time,
                'student_latency_ms': avg_student_time,
            }
        
        # Build EvaluationReport
        core_metrics = {
            'accuracy': student_acc,
            'teacher_accuracy': teacher_acc,
            'prediction_agreement': prediction_agreement,
            'accuracy_retention': student_acc / teacher_acc if teacher_acc > 0 else 0,
            'accuracy_gap': teacher_acc - student_acc,
            **student_metrics
        }
        
        per_class_metrics = {}
        if 'precision_per_class' in student_metrics:
            per_class_metrics['precision'] = student_metrics['precision_per_class']
        if 'recall_per_class' in student_metrics:
            per_class_metrics['recall'] = student_metrics['recall_per_class']
        if 'f1_per_class' in student_metrics:
            per_class_metrics['f1'] = student_metrics['f1_per_class']
            
        dist_metrics = {
            'kl_divergence': float(np.mean(kl_divs)),
            'js_divergence': float(np.mean(js_divs)),
            'confidence_correlation': float(np.mean(confidence_corrs)),
            'dei': dei_result['dei'],
            'cas': cas_result['cas']
        }
        
        metadata = {
            'teacher_params': teacher_params,
            'student_params': student_params,
            'compression_ratio': teacher_params / student_params,
            'speedup': perf_comparison.get('speedup', 0),
            'teacher_latency_ms': perf_comparison.get('teacher_latency_ms', 0),
            'student_latency_ms': perf_comparison.get('student_latency_ms', 0),
        }
        
        report = EvaluationReport(
            loss=None,
            metrics=core_metrics,
            diagnostics={},
            runtime={
                'teacher_latency_ms': perf_comparison.get('teacher_latency_ms', 0),
                'student_latency_ms': perf_comparison.get('student_latency_ms', 0),
                'speedup': perf_comparison.get('speedup', 0),
            },
            calibration=None,
            explainability=None,
            model_name=self.student.__class__.__name__,
            modality=self.modality,
            task_type='classification',
            per_class_metrics=per_class_metrics if per_class_metrics else None,
            distillation_metrics=dist_metrics,
            metadata=metadata,
        )
        
        print("[DUAL EVAL] Evaluation complete!")
        print(f"  Teacher Acc: {teacher_acc:.4f} | Student Acc: {student_acc:.4f}")
        print(f"  Agreement: {prediction_agreement:.4f} | DEI: {dei_result['dei']:.4f}")
        print(f"  CAS: {cas_result['cas']:.4f} | Speedup: {perf_comparison.get('speedup', 0):.2f}x")
        
        return report
    
    def quick_eval(self, max_batches: Optional[int] = None) -> EvaluationReport:
        """
        Quick evaluation on subset of data for rapid feedback.
        
        Args:
            max_batches: Maximum number of batches to evaluate (None = all)
            
        Returns:
            Evaluation results
        """
        original_dataloader = self.dataloader
        
        if max_batches is not None:
            # Create subset
            from itertools import islice
            self.dataloader = list(islice(self.dataloader, max_batches))
        
        results = self.evaluate(profile_performance=False)
        
        self.dataloader = original_dataloader
        return results


class CurriculumEvaluator:
    """
    Evaluate model under varied difficulty levels.
    Tests generalization and robustness.
    """
    
    def __init__(self, model: nn.Module, device: str):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    def evaluate_by_difficulty(self, dataloader_dict: Dict[str, Any]) -> Dict:
        """
        Evaluate on different difficulty tiers.
        
        Args:
            dataloader_dict: Dictionary with keys like 'easy', 'medium', 'hard'
                            mapped to respective dataloaders
                            
        Returns:
            Results per difficulty level
        """
        results = {}
        
        for difficulty, dataloader in dataloader_dict.items():
            print(f"[CURRICULUM EVAL] Evaluating on {difficulty} samples...")
            
            all_preds = []
            all_labels = []
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch in dataloader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch.get('attention_mask')
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
                    
                    preds = torch.argmax(logits, dim=-1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
                    
                    all_preds.extend(preds.cpu().numpy().tolist())
                    all_labels.extend(labels.cpu().numpy().tolist())
            
            accuracy = correct / total if total > 0 else 0
            metrics = compute_all_metrics(all_preds, all_labels)
            
            results[difficulty] = {
                'accuracy': accuracy,
                'metrics': metrics,
                'num_samples': total
            }
            
            print(f"  {difficulty.capitalize()} Accuracy: {accuracy:.4f}")
        
        # Compute robustness score (variance across difficulties)
        accuracies = [r['accuracy'] for r in results.values()]
        robustness_score = 1.0 - np.std(accuracies)
        
        results['summary'] = {
            'robustness_score': robustness_score,
            'mean_accuracy': np.mean(accuracies),
            'accuracy_variance': np.var(accuracies)
        }
        
        return results


if __name__ == "__main__":
    print("Zynthe EvalX - Enhanced Evaluator Module")
    print("="*60)
    print("Features:")
    print("  ✓ Dual-model evaluation (teacher + student)")
    print("  ✓ Extended distillation metrics")
    print("  ✓ Performance profiling")
    print("  ✓ Compression-Aware Score (CAS)")
    print("  ✓ Distillation Efficacy Index (DEI)")
    print("  ✓ Curriculum-based testing")
