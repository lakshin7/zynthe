#!/usr/bin/env python3
"""
Quick diagnostic to check if the evaluation is running properly
and producing different metrics each epoch.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import json
from core.config.config_manager import ConfigManager
from data.dataset import create_dataloaders
from core.models import load_teacher_student_models

def main():
    print("="*80)
    print("EVALUATION DIAGNOSTIC TEST")
    print("="*80)
    
    # Load config
    config_manager = ConfigManager()
    config = config_manager.config
    
    # Set device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    # Load models
    print("\nLoading models...")
    teacher, student, tokenizer = load_teacher_student_models(config, device)
    student.eval()
    
    # Load data
    print("Loading data...")
    train_loader, val_loader = create_dataloaders(config, tokenizer)
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_loader.dataset) if hasattr(train_loader, 'dataset') else 'Unknown'}")
    print(f"  Val: {len(val_loader.dataset) if hasattr(val_loader, 'dataset') else 'Unknown'}")
    print(f"  Val batches: {len(val_loader)}")
    
    # Run evaluation twice to check if results are identical
    print("\n" + "="*80)
    print("RUNNING EVALUATION #1")
    print("="*80)
    
    eval_results_1 = run_simple_eval(student, val_loader, device)
    
    print("\n" + "="*80)
    print("RUNNING EVALUATION #2 (should be identical if no training)")
    print("="*80)
    
    eval_results_2 = run_simple_eval(student, val_loader, device)
    
    # Compare results
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    
    print(f"\nEval #1 Accuracy: {eval_results_1['accuracy']:.4f}")
    print(f"Eval #2 Accuracy: {eval_results_2['accuracy']:.4f}")
    print(f"\nIdentical: {eval_results_1['accuracy'] == eval_results_2['accuracy']}")
    
    if eval_results_1['accuracy'] == eval_results_2['accuracy']:
        print("✅ Results are correctly identical (no training happened)")
    else:
        print("❌ Results differ unexpectedly!")
    
    # Save results
    results = {
        'eval_1': eval_results_1,
        'eval_2': eval_results_2,
        'identical': eval_results_1['accuracy'] == eval_results_2['accuracy']
    }
    
    with open('eval_diagnostic.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: eval_diagnostic.json")

def run_simple_eval(model, dataloader, device):
    """Simple evaluation function."""
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Move to device
            batch = {k: v.to(device) for k, v in batch.items() if hasattr(v, 'to')}
            
            # Forward pass
            outputs = model(**batch)
            
            # Get predictions
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs['logits']
            preds = torch.argmax(logits, dim=-1)
            
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(batch['labels'].cpu().numpy().tolist())
            
            if hasattr(outputs, 'loss') and outputs.loss is not None:
                total_loss += outputs.loss.item()
                num_batches += 1
            
            # Log progress
            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx + 1}/{len(dataloader)}: {len(all_preds)} predictions")
    
    # Compute accuracy
    correct = sum(p == l for p, l in zip(all_preds, all_labels))
    accuracy = correct / len(all_labels) if all_labels else 0.0
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    print(f"\nResults:")
    print(f"  Total predictions: {len(all_preds)}")
    print(f"  Correct: {correct}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Avg Loss: {avg_loss:.4f}")
    
    return {
        'accuracy': accuracy,
        'loss': avg_loss,
        'total_preds': len(all_preds),
        'correct': correct
    }

if __name__ == '__main__':
    main()
