import torch
import evaluation.metrics

class Evaluator:
    def __init__(self, model, dataloader, tokenizer, device, loss_fn=None, task_type='classification', explainer=None):
        """
        task_type: 'classification', 'multi_label', or 'regression'
        """
        self.model = model
        self.dataloader = dataloader
        self.tokenizer = tokenizer
        self.device = device
        self.loss_fn = loss_fn
        self.task_type = task_type
        self.explainer = explainer

    def evaluate(self):
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        all_preds = []
        all_labels = []

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