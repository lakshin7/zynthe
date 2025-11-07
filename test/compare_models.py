#!/usr/bin/env python3
"""
Model Comparison Script
Compares teacher and student model performance on test data with visualizations.
"""

import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
from evaluation.metrics import compute_all_metrics
from core.config.config_manager import ConfigManager

class ModelComparator:
    def __init__(self, config_path="configs/default.yaml", test_data_path="data/sample_val.jsonl"):
        self.config_manager = ConfigManager(config_path=config_path)
        self.device = self.config_manager.device()
        self.test_data_path = test_data_path
        
        # Load tokenizer
        self.tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
        
        # Load test data
        self.test_data = self.load_test_data()
        
    def load_test_data(self):
        """Load test data from JSONL file."""
        data = []
        with open(self.test_data_path, 'r') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data
    
    def prepare_batch(self, texts, labels, batch_size=8):
        """Prepare batches for model evaluation."""
        batches = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_labels = labels[i:i+batch_size]
            
            # Tokenize
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            batch = {
                'input_ids': encoded['input_ids'],
                'attention_mask': encoded['attention_mask'],
                'labels': torch.tensor(batch_labels)
            }
            batches.append(batch)
        return batches
    
    def evaluate_model(self, model, model_name):
        """Evaluate a model on test data."""
        model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        texts = [item['text'] for item in self.test_data]
        labels = [item['label'] for item in self.test_data]
        
        batches = self.prepare_batch(texts, labels)
        
        with torch.no_grad():
            for batch in batches:
                try:
                    # Move to device
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    
                    # Forward pass
                    outputs = model(**batch)
                    logits = outputs.logits
                    
                    # Get predictions
                    probs = torch.softmax(logits, dim=-1)
                    preds = torch.argmax(logits, dim=-1)
                    
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(batch['labels'].cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())
                    
                except Exception as e:
                    print(f"Error evaluating {model_name}: {e}")
                    # Fallback to CPU
                    batch = {k: v.cpu() for k, v in batch.items()}
                    outputs = model(**batch)
                    logits = outputs.logits
                    probs = torch.softmax(logits, dim=-1)
                    preds = torch.argmax(logits, dim=-1)
                    
                    all_preds.extend(preds.numpy())
                    all_labels.extend(batch['labels'].numpy())
                    all_probs.extend(probs.numpy())
        
        # Compute metrics
        metrics = compute_all_metrics(all_preds, all_labels)
        
        return {
            'model_name': model_name,
            'predictions': all_preds,
            'labels': all_labels,
            'probabilities': all_probs,
            'metrics': metrics
        }
    
    def load_teacher_model(self):
        """Load the teacher model."""
        model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            num_labels=2
        )
        model.to(self.device)
        return model
    
    def load_student_model(self, experiment_dir):
        """Load the trained student model."""
        student_path = Path(experiment_dir) / "student_model"
        if not student_path.exists():
            raise FileNotFoundError(f"Student model not found at {student_path}")
        
        model = DistilBertForSequenceClassification.from_pretrained(str(student_path))
        model.to(self.device)
        return model
    
    def create_comparison_visualization(self, teacher_results, student_results, save_path="model_comparison.png"):
        """Create comprehensive comparison visualization."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Teacher vs Student Model Comparison', fontsize=16, fontweight='bold')
        
        # 1. Accuracy Comparison
        teacher_acc = teacher_results['metrics'].get('accuracy', 0)
        student_acc = student_results['metrics'].get('accuracy', 0)
        
        axes[0, 0].bar(['Teacher', 'Student'], [teacher_acc, student_acc], 
                      color=['skyblue', 'lightcoral'], alpha=0.7)
        axes[0, 0].set_title('Accuracy Comparison')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_ylim(0, 1)
        
        # Add value labels on bars
        for i, v in enumerate([teacher_acc, student_acc]):
            axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # 2. F1 Score Comparison
        teacher_f1 = teacher_results['metrics'].get('f1', 0)
        student_f1 = student_results['metrics'].get('f1', 0)
        
        axes[0, 1].bar(['Teacher', 'Student'], [teacher_f1, student_f1], 
                      color=['skyblue', 'lightcoral'], alpha=0.7)
        axes[0, 1].set_title('F1 Score Comparison')
        axes[0, 1].set_ylabel('F1 Score')
        axes[0, 1].set_ylim(0, 1)
        
        for i, v in enumerate([teacher_f1, student_f1]):
            axes[0, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # 3. Precision vs Recall
        teacher_prec = teacher_results['metrics'].get('precision', 0)
        teacher_rec = teacher_results['metrics'].get('recall', 0)
        student_prec = student_results['metrics'].get('precision', 0)
        student_rec = student_results['metrics'].get('recall', 0)
        
        x = np.arange(2)
        width = 0.35
        
        axes[0, 2].bar(x - width/2, [teacher_prec, teacher_rec], width, 
                      label='Teacher', color='skyblue', alpha=0.7)
        axes[0, 2].bar(x + width/2, [student_prec, student_rec], width, 
                      label='Student', color='lightcoral', alpha=0.7)
        axes[0, 2].set_title('Precision vs Recall')
        axes[0, 2].set_ylabel('Score')
        axes[0, 2].set_xticks(x)
        axes[0, 2].set_xticklabels(['Precision', 'Recall'])
        axes[0, 2].legend()
        axes[0, 2].set_ylim(0, 1)
        
        # 4. Confusion Matrix - Teacher
        from sklearn.metrics import confusion_matrix
        cm_teacher = confusion_matrix(teacher_results['labels'], teacher_results['predictions'])
        sns.heatmap(cm_teacher, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
        axes[1, 0].set_title('Teacher Confusion Matrix')
        axes[1, 0].set_xlabel('Predicted')
        axes[1, 0].set_ylabel('Actual')
        
        # 5. Confusion Matrix - Student
        cm_student = confusion_matrix(student_results['labels'], student_results['predictions'])
        sns.heatmap(cm_student, annot=True, fmt='d', cmap='Reds', ax=axes[1, 1])
        axes[1, 1].set_title('Student Confusion Matrix')
        axes[1, 1].set_xlabel('Predicted')
        axes[1, 1].set_ylabel('Actual')
        
        # 6. Performance Summary Table
        axes[1, 2].axis('off')
        
        # Create summary table
        summary_data = [
            ['Metric', 'Teacher', 'Student', 'Difference'],
            ['Accuracy', f'{teacher_acc:.3f}', f'{student_acc:.3f}', f'{student_acc - teacher_acc:+.3f}'],
            ['F1 Score', f'{teacher_f1:.3f}', f'{student_f1:.3f}', f'{student_f1 - teacher_f1:+.3f}'],
            ['Precision', f'{teacher_prec:.3f}', f'{student_prec:.3f}', f'{student_prec - teacher_prec:+.3f}'],
            ['Recall', f'{teacher_rec:.3f}', f'{student_rec:.3f}', f'{student_rec - teacher_rec:+.3f}']
        ]
        
        table = axes[1, 2].table(cellText=summary_data[1:], colLabels=summary_data[0],
                               cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        axes[1, 2].set_title('Performance Summary', pad=20)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to free memory
        
        return save_path
    
    def print_detailed_results(self, teacher_results, student_results):
        """Print detailed comparison results."""
        print("\n" + "="*80)
        print("DETAILED MODEL COMPARISON RESULTS")
        print("="*80)
        
        print(f"\n📊 TEACHER MODEL PERFORMANCE:")
        print(f"   Accuracy:  {teacher_results['metrics'].get('accuracy', 0):.4f}")
        print(f"   F1 Score:  {teacher_results['metrics'].get('f1', 0):.4f}")
        print(f"   Precision: {teacher_results['metrics'].get('precision', 0):.4f}")
        print(f"   Recall:    {teacher_results['metrics'].get('recall', 0):.4f}")
        
        print(f"\n🎓 STUDENT MODEL PERFORMANCE:")
        print(f"   Accuracy:  {student_results['metrics'].get('accuracy', 0):.4f}")
        print(f"   F1 Score:  {student_results['metrics'].get('f1', 0):.4f}")
        print(f"   Precision: {student_results['metrics'].get('precision', 0):.4f}")
        print(f"   Recall:    {student_results['metrics'].get('recall', 0):.4f}")
        
        print(f"\n📈 PERFORMANCE DIFFERENCES (Student - Teacher):")
        acc_diff = student_results['metrics'].get('accuracy', 0) - teacher_results['metrics'].get('accuracy', 0)
        f1_diff = student_results['metrics'].get('f1', 0) - teacher_results['metrics'].get('f1', 0)
        prec_diff = student_results['metrics'].get('precision', 0) - teacher_results['metrics'].get('precision', 0)
        rec_diff = student_results['metrics'].get('recall', 0) - teacher_results['metrics'].get('recall', 0)
        
        print(f"   Accuracy:  {acc_diff:+.4f}")
        print(f"   F1 Score:  {f1_diff:+.4f}")
        print(f"   Precision: {prec_diff:+.4f}")
        print(f"   Recall:    {rec_diff:+.4f}")
        
        # Knowledge retention analysis
        retention_rate = (student_results['metrics'].get('accuracy', 0) / 
                         max(teacher_results['metrics'].get('accuracy', 0), 0.001)) * 100
        print(f"\n🧠 KNOWLEDGE RETENTION: {retention_rate:.1f}%")
        
        if retention_rate >= 95:
            print("   ✅ Excellent knowledge retention!")
        elif retention_rate >= 90:
            print("   ✅ Good knowledge retention!")
        elif retention_rate >= 80:
            print("   ⚠️  Moderate knowledge retention")
        else:
            print("   ❌ Poor knowledge retention - consider adjusting distillation parameters")
    
    def run_comparison(self, experiment_dir):
        """Run complete model comparison."""
        print("🔍 Starting Model Comparison...")
        print(f"📁 Using experiment directory: {experiment_dir}")
        print(f"📊 Test data: {self.test_data_path}")
        print(f"🔢 Test samples: {len(self.test_data)}")
        
        # Load models
        print("\n📥 Loading models...")
        teacher_model = self.load_teacher_model()
        student_model = self.load_student_model(experiment_dir)
        
        # Evaluate models
        print("\n🧪 Evaluating Teacher Model...")
        teacher_results = self.evaluate_model(teacher_model, "Teacher")
        
        print("🧪 Evaluating Student Model...")
        student_results = self.evaluate_model(student_model, "Student")
        
        # Print results
        self.print_detailed_results(teacher_results, student_results)
        
        # Create visualization
        print("\n📊 Creating comparison visualization...")
        viz_path = self.create_comparison_visualization(teacher_results, student_results)
        print(f"📈 Visualization saved to: {viz_path}")
        
        return teacher_results, student_results

def main():
    """Main function to run model comparison."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare Teacher and Student Models")
    parser.add_argument("--experiment-dir", type=str, required=True,
                       help="Path to experiment directory containing trained student model")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                       help="Path to config file")
    parser.add_argument("--test-data", type=str, default="data/sample_val.jsonl",
                       help="Path to test data")
    
    args = parser.parse_args()
    
    # Run comparison
    comparator = ModelComparator(args.config, args.test_data)
    teacher_results, student_results = comparator.run_comparison(args.experiment_dir)
    
    print("\n✅ Model comparison completed successfully!")

if __name__ == "__main__":
    main()
