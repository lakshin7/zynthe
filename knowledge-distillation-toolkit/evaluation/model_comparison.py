"""
Model Comparison Module
Compare Teacher and Student models side-by-side with comprehensive metrics and visualizations.
"""

import torch
import json
import os
from pathlib import Path
from typing import Dict, Tuple, Optional
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


class ModelComparator:
    """Compare Teacher and Student models with comprehensive analysis."""
    
    def __init__(
        self, 
        teacher_path: str,
        student_path: str,
        device: str = "mps",
        use_same_tokenizer: bool = True
    ):
        """
        Initialize model comparator.
        
        Args:
            teacher_path: Path to teacher model directory
            student_path: Path to student model directory
            device: Device to run models on (mps, cuda, cpu)
            use_same_tokenizer: Use student tokenizer for both (recommended)
        """
        self.teacher_path = Path(teacher_path)
        self.student_path = Path(student_path)
        self.device = device
        
        print(f"🔍 Loading models on device: {device}")
        
        # Load tokenizer (using student tokenizer for consistency)
        tokenizer_path = student_path if use_same_tokenizer else teacher_path
        print(f"📝 Loading tokenizer from: {tokenizer_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        # Load models
        print(f"👨‍🏫 Loading Teacher model from: {teacher_path}")
        self.teacher = AutoModelForSequenceClassification.from_pretrained(teacher_path)
        self.teacher.to(device)
        self.teacher.eval()
        
        print(f"👨‍🎓 Loading Student model from: {student_path}")
        self.student = AutoModelForSequenceClassification.from_pretrained(student_path)
        self.student.to(device)
        self.student.eval()
        
        # Model statistics
        self.teacher_params = sum(p.numel() for p in self.teacher.parameters())
        self.student_params = sum(p.numel() for p in self.student.parameters())
        self.compression_ratio = self.teacher_params / self.student_params
        
        print(f"\n📊 Model Statistics:")
        print(f"   Teacher: {self.teacher_params:,} parameters")
        print(f"   Student: {self.student_params:,} parameters")
        print(f"   Compression: {self.compression_ratio:.2f}x smaller")
        
    def evaluate_model(
        self, 
        model: torch.nn.Module,
        dataloader: DataLoader,
        model_name: str = "Model"
    ) -> Dict:
        """
        Evaluate a single model on the given dataloader.
        
        Args:
            model: The model to evaluate
            dataloader: DataLoader with evaluation data
            model_name: Name for logging
            
        Returns:
            Dictionary with all metrics
        """
        model.eval()
        all_predictions = []
        all_labels = []
        all_logits = []
        total_loss = 0.0
        
        print(f"\n🔬 Evaluating {model_name}...")
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Evaluating {model_name}"):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                # Get predictions
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                
                # Store results
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_logits.extend(logits.cpu().numpy())
                total_loss += outputs.loss.item()
        
        # Convert to numpy
        predictions = np.array(all_predictions)
        labels = np.array(all_labels)
        logits = np.array(all_logits)
        
        # Compute metrics
        accuracy = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions, average='macro', zero_division=0)
        recall = recall_score(labels, predictions, average='macro', zero_division=0)
        f1 = f1_score(labels, predictions, average='macro', zero_division=0)
        
        # Per-class metrics
        precision_per_class = precision_score(labels, predictions, average=None, zero_division=0)
        recall_per_class = recall_score(labels, predictions, average=None, zero_division=0)
        f1_per_class = f1_score(labels, predictions, average=None, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        
        # Classification report
        class_report = classification_report(labels, predictions, output_dict=True)
        
        avg_loss = total_loss / len(dataloader)
        
        results = {
            'model_name': model_name,
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'loss': float(avg_loss),
            'precision_per_class': precision_per_class.tolist(),
            'recall_per_class': recall_per_class.tolist(),
            'f1_per_class': f1_per_class.tolist(),
            'confusion_matrix': cm.tolist(),
            'classification_report': class_report,
            'predictions': predictions.tolist(),
            'labels': labels.tolist(),
            'logits': logits.tolist(),
            'num_parameters': sum(p.numel() for p in model.parameters())
        }
        
        print(f"\n✅ {model_name} Results:")
        print(f"   Accuracy:  {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall:    {recall:.4f}")
        print(f"   F1-Score:  {f1:.4f}")
        print(f"   Avg Loss:  {avg_loss:.4f}")
        
        return results
    
    def compare_models(self, dataloader: DataLoader) -> Tuple[Dict, Dict]:
        """
        Compare both teacher and student models.
        
        Args:
            dataloader: DataLoader with evaluation data
            
        Returns:
            Tuple of (teacher_results, student_results)
        """
        print("\n" + "="*60)
        print("🎯 TEACHER vs STUDENT COMPARISON")
        print("="*60)
        
        # Evaluate teacher
        teacher_results = self.evaluate_model(
            self.teacher, 
            dataloader, 
            "Teacher Model"
        )
        
        # Evaluate student
        student_results = self.evaluate_model(
            self.student, 
            dataloader, 
            "Student Model"
        )
        
        # Add compression info
        teacher_results['compression_ratio'] = 1.0
        student_results['compression_ratio'] = self.compression_ratio
        
        return teacher_results, student_results
    
    def visualize_comparison(
        self,
        teacher_results: Dict,
        student_results: Dict,
        save_dir: str,
        show_plots: bool = False
    ):
        """
        Create comprehensive visualizations comparing models.
        
        Args:
            teacher_results: Teacher model evaluation results
            student_results: Student model evaluation results
            save_dir: Directory to save plots
            show_plots: Whether to display plots
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n📊 Creating comparison visualizations...")
        
        # 1. Metrics Bar Chart
        self._plot_metrics_comparison(teacher_results, student_results, save_dir)
        
        # 2. Confusion Matrices Side-by-Side
        self._plot_confusion_matrices(teacher_results, student_results, save_dir)
        
        # 3. Per-Class Performance
        self._plot_per_class_metrics(teacher_results, student_results, save_dir)
        
        # 4. Model Size vs Performance
        self._plot_efficiency_chart(teacher_results, student_results, save_dir)
        
        # 5. Detailed Comparison Table
        self._create_comparison_table(teacher_results, student_results, save_dir)
        
        print(f"✅ Visualizations saved to: {save_dir}")
        
        if not show_plots:
            plt.close('all')
    
    def _plot_metrics_comparison(self, teacher_res: Dict, student_res: Dict, save_dir: Path):
        """Plot main metrics comparison bar chart."""
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        teacher_vals = [teacher_res[m] for m in metrics]
        student_vals = [student_res[m] for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bars1 = ax.bar(x - width/2, teacher_vals, width, label='Teacher', color='#2E86AB', alpha=0.8)
        bars2 = ax.bar(x + width/2, student_vals, width, label='Student', color='#A23B72', alpha=0.8)
        
        ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Teacher vs Student: Performance Comparison', fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels([m.capitalize() for m in metrics])
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim([0, 1.0])
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'metrics_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Saved: metrics_comparison.png")
    
    def _plot_confusion_matrices(self, teacher_res: Dict, student_res: Dict, save_dir: Path):
        """Plot confusion matrices side-by-side."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Teacher confusion matrix
        cm_teacher = np.array(teacher_res['confusion_matrix'])
        sns.heatmap(cm_teacher, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                   cbar_kws={'label': 'Count'})
        axes[0].set_title('Teacher Model\nConfusion Matrix', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Predicted', fontsize=11)
        axes[0].set_ylabel('Actual', fontsize=11)
        
        # Student confusion matrix
        cm_student = np.array(student_res['confusion_matrix'])
        sns.heatmap(cm_student, annot=True, fmt='d', cmap='Purples', ax=axes[1],
                   cbar_kws={'label': 'Count'})
        axes[1].set_title('Student Model\nConfusion Matrix', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Predicted', fontsize=11)
        axes[1].set_ylabel('Actual', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'confusion_matrices_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Saved: confusion_matrices_comparison.png")
    
    def _plot_per_class_metrics(self, teacher_res: Dict, student_res: Dict, save_dir: Path):
        """Plot per-class performance metrics."""
        num_classes = len(teacher_res['f1_per_class'])
        metrics = ['precision_per_class', 'recall_per_class', 'f1_per_class']
        metric_names = ['Precision', 'Recall', 'F1-Score']
        
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
            teacher_vals = teacher_res[metric]
            student_vals = student_res[metric]
            
            x = np.arange(num_classes)
            width = 0.35
            
            axes[idx].bar(x - width/2, teacher_vals, width, label='Teacher', color='#2E86AB', alpha=0.8)
            axes[idx].bar(x + width/2, student_vals, width, label='Student', color='#A23B72', alpha=0.8)
            
            axes[idx].set_xlabel('Class', fontsize=11, fontweight='bold')
            axes[idx].set_ylabel(name, fontsize=11, fontweight='bold')
            axes[idx].set_title(f'{name} per Class', fontsize=12, fontweight='bold')
            axes[idx].set_xticks(x)
            axes[idx].set_xticklabels([f'Class {i}' for i in range(num_classes)])
            axes[idx].legend()
            axes[idx].grid(axis='y', alpha=0.3, linestyle='--')
            axes[idx].set_ylim([0, 1.0])
        
        plt.tight_layout()
        plt.savefig(save_dir / 'per_class_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Saved: per_class_comparison.png")
    
    def _plot_efficiency_chart(self, teacher_res: Dict, student_res: Dict, save_dir: Path):
        """Plot model efficiency (size vs performance)."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        models = ['Teacher', 'Student']
        params = [teacher_res['num_parameters'] / 1e6, student_res['num_parameters'] / 1e6]  # in millions
        accuracies = [teacher_res['accuracy'], student_res['accuracy']]
        colors = ['#2E86AB', '#A23B72']
        
        # Scatter plot with size representing accuracy
        scatter = ax.scatter(params, accuracies, s=[a*1000 for a in accuracies], 
                           c=colors, alpha=0.6, edgecolors='black', linewidth=2)
        
        # Add labels
        for i, (model, param, acc) in enumerate(zip(models, params, accuracies)):
            ax.annotate(f'{model}\n{param:.1f}M params\nAcc: {acc:.3f}',
                       xy=(param, acc), xytext=(10, 10),
                       textcoords='offset points', fontsize=10,
                       bbox=dict(boxstyle='round,pad=0.5', fc=colors[i], alpha=0.3))
        
        ax.set_xlabel('Model Size (Million Parameters)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax.set_title('Model Efficiency: Size vs Performance', fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim([min(accuracies) - 0.05, 1.0])
        
        # Add compression annotation
        compression = teacher_res['num_parameters'] / student_res['num_parameters']
        accuracy_drop = (teacher_res['accuracy'] - student_res['accuracy']) * 100
        ax.text(0.05, 0.95, 
               f'Compression: {compression:.2f}x\nAccuracy Drop: {accuracy_drop:.2f}%',
               transform=ax.transAxes, fontsize=11,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(save_dir / 'efficiency_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Saved: efficiency_comparison.png")
    
    def _create_comparison_table(self, teacher_res: Dict, student_res: Dict, save_dir: Path):
        """Create detailed comparison table as image."""
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare data
        metrics_data = [
            ['Metric', 'Teacher', 'Student', 'Difference'],
            ['Parameters', f"{teacher_res['num_parameters']:,}", f"{student_res['num_parameters']:,}", 
             f"{teacher_res['compression_ratio']:.2f}x smaller"],
            ['Accuracy', f"{teacher_res['accuracy']:.4f}", f"{student_res['accuracy']:.4f}",
             f"{(teacher_res['accuracy'] - student_res['accuracy']):.4f}"],
            ['Precision', f"{teacher_res['precision']:.4f}", f"{student_res['precision']:.4f}",
             f"{(teacher_res['precision'] - student_res['precision']):.4f}"],
            ['Recall', f"{teacher_res['recall']:.4f}", f"{student_res['recall']:.4f}",
             f"{(teacher_res['recall'] - student_res['recall']):.4f}"],
            ['F1-Score', f"{teacher_res['f1']:.4f}", f"{student_res['f1']:.4f}",
             f"{(teacher_res['f1'] - student_res['f1']):.4f}"],
            ['Avg Loss', f"{teacher_res['loss']:.4f}", f"{student_res['loss']:.4f}",
             f"{(teacher_res['loss'] - student_res['loss']):.4f}"],
        ]
        
        table = ax.table(cellText=metrics_data, cellLoc='center', loc='center',
                        colWidths=[0.25, 0.25, 0.25, 0.25])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header row
        for i in range(4):
            table[(0, i)].set_facecolor('#2E86AB')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Style data rows
        for i in range(1, len(metrics_data)):
            for j in range(4):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#E8F4F8')
        
        plt.title('Teacher vs Student: Detailed Comparison', 
                 fontsize=14, fontweight='bold', pad=20)
        
        plt.savefig(save_dir / 'comparison_table.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Saved: comparison_table.png")
    
    def save_results(
        self,
        teacher_results: Dict,
        student_results: Dict,
        save_dir: str
    ):
        """Save comparison results as JSON."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare summary
        summary = {
            'teacher': {
                k: v for k, v in teacher_results.items() 
                if k not in ['predictions', 'labels', 'logits']  # Exclude large arrays
            },
            'student': {
                k: v for k, v in student_results.items()
                if k not in ['predictions', 'labels', 'logits']
            },
            'comparison': {
                'accuracy_diff': teacher_results['accuracy'] - student_results['accuracy'],
                'f1_diff': teacher_results['f1'] - student_results['f1'],
                'compression_ratio': student_results['compression_ratio'],
                'parameter_reduction': (1 - 1/student_results['compression_ratio']) * 100,
            }
        }
        
        # Save JSON
        with open(save_dir / 'comparison_results.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n💾 Results saved to: {save_dir / 'comparison_results.json'}")
        
    def generate_report(
        self,
        teacher_results: Dict,
        student_results: Dict,
        save_dir: str
    ):
        """Generate a comprehensive markdown report."""
        save_dir = Path(save_dir)
        
        report = []
        report.append("# 📊 Teacher vs Student Model Comparison Report\n")
        report.append(f"Generated on: {Path(save_dir).name}\n")
        report.append("---\n\n")
        
        # Executive Summary
        report.append("## 📋 Executive Summary\n\n")
        compression = student_results['compression_ratio']
        acc_diff = (teacher_results['accuracy'] - student_results['accuracy']) * 100
        f1_diff = (teacher_results['f1'] - student_results['f1']) * 100
        
        report.append(f"- **Compression Ratio**: {compression:.2f}x smaller\n")
        report.append(f"- **Parameter Reduction**: {(1 - 1/compression) * 100:.1f}%\n")
        report.append(f"- **Accuracy Drop**: {acc_diff:.2f}%\n")
        report.append(f"- **F1-Score Drop**: {f1_diff:.2f}%\n\n")
        
        # Model Statistics
        report.append("## 🏗️ Model Statistics\n\n")
        report.append("| Metric | Teacher | Student |\n")
        report.append("|--------|---------|----------|\n")
        report.append(f"| Parameters | {teacher_results['num_parameters']:,} | {student_results['num_parameters']:,} |\n")
        report.append(f"| Size Ratio | 1.0x | {1/compression:.3f}x |\n\n")
        
        # Performance Metrics
        report.append("## 📈 Performance Metrics\n\n")
        report.append("| Metric | Teacher | Student | Difference |\n")
        report.append("|--------|---------|---------|------------|\n")
        report.append(f"| Accuracy | {teacher_results['accuracy']:.4f} | {student_results['accuracy']:.4f} | {teacher_results['accuracy'] - student_results['accuracy']:.4f} |\n")
        report.append(f"| Precision | {teacher_results['precision']:.4f} | {student_results['precision']:.4f} | {teacher_results['precision'] - student_results['precision']:.4f} |\n")
        report.append(f"| Recall | {teacher_results['recall']:.4f} | {student_results['recall']:.4f} | {teacher_results['recall'] - student_results['recall']:.4f} |\n")
        report.append(f"| F1-Score | {teacher_results['f1']:.4f} | {student_results['f1']:.4f} | {teacher_results['f1'] - student_results['f1']:.4f} |\n\n")
        
        # Visualizations
        report.append("## 📊 Visualizations\n\n")
        report.append("### Metrics Comparison\n")
        report.append("![Metrics Comparison](metrics_comparison.png)\n\n")
        report.append("### Confusion Matrices\n")
        report.append("![Confusion Matrices](confusion_matrices_comparison.png)\n\n")
        report.append("### Per-Class Performance\n")
        report.append("![Per-Class Metrics](per_class_comparison.png)\n\n")
        report.append("### Efficiency Analysis\n")
        report.append("![Efficiency](efficiency_comparison.png)\n\n")
        
        # Conclusion
        report.append("## 🎯 Conclusion\n\n")
        if abs(acc_diff) < 2.0:
            verdict = "✅ **Excellent**: Student model maintains near-identical performance with significant size reduction."
        elif abs(acc_diff) < 5.0:
            verdict = "✔️ **Good**: Student model shows acceptable performance with good compression ratio."
        else:
            verdict = "⚠️ **Fair**: Student model shows noticeable performance drop. Consider retraining with adjusted hyperparameters."
        
        report.append(f"{verdict}\n\n")
        report.append(f"The student model achieves **{compression:.2f}x compression** ")
        report.append(f"with only **{abs(acc_diff):.2f}%** accuracy difference, ")
        report.append(f"making it a {('viable' if abs(acc_diff) < 3 else 'reasonable')} ")
        report.append(f"candidate for deployment in resource-constrained environments.\n")
        
        # Save report
        report_path = save_dir / 'COMPARISON_REPORT.md'
        with open(report_path, 'w') as f:
            f.writelines(report)
        
        print(f"\n📝 Report generated: {report_path}")
