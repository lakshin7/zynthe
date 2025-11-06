import torch
import evaluation.metrics
from typing import Optional, Callable, Dict, Any

class Evaluator:
    def __init__(
        self, 
        model, 
        dataloader, 
        tokenizer, 
        device, 
        loss_fn=None, 
        task_type='classification', 
        explainer=None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        update_frequency: int = 10
    ):
        """
        task_type: 'classification', 'multi_label', or 'regression'
        progress_callback: Optional callback for real-time progress updates (WebSocket streaming)
        update_frequency: Update progress every N batches (default: 10)
        """
        self.model = model
        self.dataloader = dataloader
        self.tokenizer = tokenizer
        self.device = device
        self.loss_fn = loss_fn
        self.task_type = task_type
        self.explainer = explainer
        
        # Real-time progress streaming for UI
        self.progress_callback = progress_callback
        self.update_frequency = update_frequency

    def evaluate(self):
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        all_preds = []
        all_labels = []
        
        # Calculate total batches for progress tracking
        total_batches = len(self.dataloader) if hasattr(self.dataloader, '__len__') else None
        batch_count = 0

        with torch.no_grad():
            for batch in self.dataloader:
                # Handle batch being dict or tuple/list
                if isinstance(batch, dict):
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch.get('attention_mask', None)
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(self.device)
                    labels = batch['labels'].to(self.device)
                elif isinstance(batch, (tuple, list)):
                    input_ids, labels = batch[:2]
                    input_ids = input_ids.to(self.device)
                    labels = labels.to(self.device)
                    attention_mask = batch[2].to(self.device) if len(batch) > 2 else None
                else:
                    raise ValueError("Unsupported batch format")

                try:
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                except RuntimeError as e:
                    # For PTQ or quantized models on MPS/CPU, fallback to CPU evaluation
                    if 'mps' in str(self.device) or 'cpu' in str(self.device):
                        outputs = self.model(input_ids=input_ids.cpu(), attention_mask=attention_mask.cpu() if attention_mask is not None else None)
                    else:
                        raise e

                logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]

                if self.loss_fn is not None:
                    if self.task_type == 'regression':
                        loss = self.loss_fn(logits.squeeze(), labels.float())
                    else:
                        loss = self.loss_fn(logits, labels)
                    total_loss += loss.item() * input_ids.size(0)
                total_samples += input_ids.size(0)

                # Predictions
                if self.task_type == 'classification':
                    preds = torch.argmax(logits, dim=-1)
                elif self.task_type == 'multi_label':
                    preds = (torch.sigmoid(logits) > 0.5).int()
                else:  # regression
                    preds = logits

                # Safely convert preds and labels to list
                if isinstance(preds, torch.Tensor):
                    preds = preds.detach().cpu().tolist()
                if isinstance(labels, torch.Tensor):
                    labels = labels.detach().cpu().tolist()

                if isinstance(preds, list):
                    all_preds.extend(preds)
                else:
                    all_preds.append(preds)
                if isinstance(labels, list):
                    all_labels.extend(labels)
                else:
                    all_labels.append(labels)
                
                batch_count += 1
                
                # ========== REAL-TIME PROGRESS STREAMING ==========
                # Send progress updates to UI via WebSocket
                if self.progress_callback and batch_count % self.update_frequency == 0:
                    # Calculate running accuracy for classification tasks
                    running_accuracy = None
                    if self.task_type in ['classification', 'multi_label'] and len(all_preds) > 0:
                        correct = sum(p == l for p, l in zip(all_preds, all_labels))
                        running_accuracy = correct / len(all_preds)
                    
                    progress_payload = {
                        'type': 'evaluation_progress',
                        'stage': 'evaluation',
                        'batch': batch_count,
                        'total_batches': total_batches,
                        'progress': (batch_count / total_batches * 100) if total_batches else None,
                        'samples_processed': total_samples,
                        'current_accuracy': running_accuracy,
                        'current_loss': (total_loss / total_samples) if total_samples > 0 and self.loss_fn else None
                    }
                    
                    try:
                        self.progress_callback(progress_payload)
                    except Exception as e:
                        print(f"[WARNING] Progress callback failed: {e}")
                # ================================================

        avg_loss = total_loss / total_samples if self.loss_fn is not None else None

        # Compute metrics only for classification tasks
        if self.task_type in ['classification', 'multi_label']:
            metrics = evaluation.metrics.compute_all_metrics(all_preds, all_labels)
            if metrics is None:
                metrics = {}
            result = {
                'loss': avg_loss,
                'accuracy': metrics.get('accuracy'),
                'f1': metrics.get('f1'),
                'precision': metrics.get('precision'),
                'recall': metrics.get('recall'),
                'all_preds': all_preds,
                'all_labels': all_labels
            }
        else:
            result = {
                'loss': avg_loss,
                'preds': all_preds,
                'labels': all_labels
            }

        if self.explainer is not None:
            self.explainer.visualize(all_preds, all_labels)
        return result

    def run_all(self):
        """
        Run comprehensive evaluation and return metrics.
        This method provides compatibility with the expected interface.
        """
        return self.evaluate()