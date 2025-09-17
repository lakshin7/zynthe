import copy
from evaluation.metrics import compute_all_metrics
from evaluation.visualizer import plot_training_curves
from evaluation.metrics import plot_metrics
from core.distillers.multi_stage_distiller import MultiStageDistiller
import torch
from torch.optim import AdamW
import os

class Trainer:
    def __init__(self, teacher, student, tokenizer, config, device, experiment_dir):
        self.teacher = teacher
        self.student = student
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        self.experiment_dir = experiment_dir
        self.optimizer = AdamW(self.student.parameters(), lr=self.config['train'].get('lr', 5e-5))
        distil_cfg = self.config['distillation']
        # Currently using MultiStageDistiller, future: can select dynamically
        self.distiller = MultiStageDistiller(
            student=self.student,
            teacher=self.teacher,
            config=self.config
        )
        # Add a compute_loss method that matches the expected interface
        self.distiller.compute_loss = self._compute_distillation_loss
        self.train_losses = []
        self.val_losses = []
        self.metrics_history = {'accuracy': [], 'f1': [], 'precision': [], 'recall': []}
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.early_stop_patience = self.config['train'].get('early_stop_patience', 2)
        self.no_improve_epochs = 0
        self.last_preds = []
        self.last_labels = []

    def train_epoch(self, dataloader):
        self.student.train()
        self.teacher.eval()
        total_loss = 0.0
        num_batches = 0
        for batch_idx, batch in enumerate(dataloader):
            if not batch or not isinstance(batch, dict):
                print(f"[WARNING] Skipping empty or malformed batch at index {batch_idx}")
                continue
            try:
                batch = {k: v.to(self.device) for k, v in batch.items() if hasattr(v, 'to')}
            except Exception as e:
                print(f"[WARNING] Failed to move batch to device at index {batch_idx}: {e}")
                continue
            with torch.no_grad():
                try:
                    teacher_outputs = self.teacher(**batch)
                except Exception as e:
                    print(f"[WARNING] Teacher forward pass failed at batch {batch_idx}: {e}")
                    continue
            try:
                student_outputs = self.student(**batch)
            except Exception as e:
                print(f"[WARNING] Student forward pass failed at batch {batch_idx}: {e}")
                continue
            labels = batch.get('labels', None)
            try:
                loss_dict = self.distiller.compute_loss(
                    student_outputs=student_outputs,
                    teacher_outputs=teacher_outputs,
                    targets=labels
                )
                # Extract total loss from dict (assuming 'total' key or sum all losses)
                if isinstance(loss_dict, dict):
                    loss = sum(loss_dict.values()) if loss_dict else torch.tensor(0.0, device=self.device)
                else:
                    loss = loss_dict
            except Exception as e:
                print(f"[WARNING] Loss computation failed at batch {batch_idx}: {e}")
                loss = torch.tensor(0.0, device=self.device)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            num_batches += 1
        avg_loss = total_loss / max(num_batches, 1)
        self.train_losses.append(avg_loss)
        print(f"[TRAIN] Epoch training completed. Average Loss: {avg_loss:.4f}")
        return avg_loss

    def evaluate(self, dataloader):
        self.student.eval()
        self.teacher.eval()
        total_loss = 0.0
        num_batches = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if not batch or not isinstance(batch, dict):
                    print(f"[WARNING] Skipping empty or malformed batch at index {batch_idx} during evaluation")
                    continue
                try:
                    batch = {k: v.to(self.device) for k, v in batch.items() if hasattr(v, 'to')}
                except Exception as e:
                    print(f"[WARNING] Failed to move batch to device at index {batch_idx} during evaluation: {e}")
                    continue
                try:
                    teacher_outputs = self.teacher(**batch)
                except Exception as e:
                    print(f"[WARNING] Teacher forward pass failed at batch {batch_idx} during evaluation: {e}")
                    continue
                try:
                    student_outputs = self.student(**batch)
                except Exception as e:
                    print(f"[WARNING] Student forward pass failed at batch {batch_idx} during evaluation: {e}")
                    continue
                labels = batch.get('labels', None)
                try:
                    loss_dict = self.distiller.compute_loss(
                        student_outputs=student_outputs,
                        teacher_outputs=teacher_outputs,
                        targets=labels
                    )
                    # Extract total loss from dict
                    if isinstance(loss_dict, dict):
                        loss = sum(loss_dict.values()) if loss_dict else torch.tensor(0.0, device=self.device)
                    else:
                        loss = loss_dict
                except Exception as e:
                    print(f"[WARNING] Loss computation failed at batch {batch_idx} during evaluation: {e}")
                    loss = torch.tensor(0.0, device=self.device)
                total_loss += loss.item()
                num_batches += 1
                if labels is not None and hasattr(student_outputs, 'logits'):
                    preds = torch.argmax(student_outputs.logits, dim=-1)
                    all_preds.extend(preds.cpu().numpy().tolist())
                    all_labels.extend(labels.cpu().numpy().tolist())
        avg_loss = total_loss / max(num_batches, 1)
        self.val_losses.append(avg_loss)
        self.last_preds = all_preds
        self.last_labels = all_labels
        metrics = []
        if all_labels and all_preds and len(all_labels) == len(all_preds):
            try:
                computed_metrics = compute_all_metrics(all_preds, all_labels)
                if not isinstance(computed_metrics, dict):
                    computed_metrics = {}
                for key in self.metrics_history.keys():
                    self.metrics_history[key].append(computed_metrics.get(key, 0))
                metrics = [computed_metrics]
            except Exception as e:
                print(f"[WARNING] Metric computation failed during evaluation: {e}")
        print(f"[EVAL] Evaluation completed. Average Loss: {avg_loss:.4f}, Metrics: {metrics if metrics else 'N/A'}")
        return avg_loss, metrics

    def fit(self, train_loader, val_loader):
        metrics_history = []
        epochs = self.config['train']['epochs']
        print(f"[INFO] Training started for {epochs} epochs.")
        for epoch in range(epochs):
            print(f"[INFO] Starting epoch {epoch+1}/{epochs}")
            train_loss = self.train_epoch(train_loader)
            val_loss, val_metrics = self.evaluate(val_loader)
            if not isinstance(val_metrics, list):
                val_metrics = []
            val_metrics_dict = val_metrics[0] if val_metrics else {}
            metrics_history.append({
                'epoch': epoch+1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                **val_metrics_dict
            })

            print(f"[INFO] Epoch {epoch+1} summary: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Metrics={val_metrics_dict}")

            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = copy.deepcopy(self.student.state_dict())
                self.no_improve_epochs = 0
                print(f"[INFO] Validation loss improved to {val_loss:.4f}. Saving best model state.")
            else:
                self.no_improve_epochs += 1
                print(f"[INFO] No improvement in validation loss for {self.no_improve_epochs} epoch(s).")
                if self.no_improve_epochs >= self.early_stop_patience:
                    print(f"[INFO] Early stopping triggered after {self.no_improve_epochs} epochs without improvement.")
                    break

        # Final summary
        self._summarize_training(metrics_history)

        # Restore best model and save
        if self.best_model_state:
            self.student.load_state_dict(self.best_model_state)
            print('[INFO] Restored best model before saving.')
            student_save_dir = os.path.join(self.experiment_dir, 'student_model')
            self.student.save_pretrained(student_save_dir)
            self.tokenizer.save_pretrained(student_save_dir)
            print(f'[INFO] Student model and tokenizer saved to {student_save_dir}')
        else:
            print('[WARNING] No best model state found to restore.')

        # Plot once at the end safely
        agg_metrics = {k: self.metrics_history[k] for k in self.metrics_history.keys()}
        save_path = os.path.join(self.experiment_dir, 'training_curves.png')
        try:
            plot_training_curves(self.train_losses, self.val_losses, agg_metrics, save_path)
            print(f"[INFO] Training curves saved to {save_path}")
        except Exception as e:
            print(f"[WARNING] Failed to plot training curves: {e}")

        if self.last_preds and self.last_labels and len(self.last_preds) == len(self.last_labels):
            try:
                final_metrics = compute_all_metrics(self.last_preds, self.last_labels)
                if isinstance(final_metrics, dict):
                    plot_metrics([final_metrics], self.experiment_dir)
                    print(f"[INFO] Final metrics plotted and saved in {self.experiment_dir}")
            except Exception as e:
                print(f"[WARNING] Failed to plot final metrics: {e}")
        else:
            print("[INFO] Skipping final metrics plotting due to missing or mismatched predictions and labels.")

    def _compute_distillation_loss(self, student_outputs, teacher_outputs, targets=None):
        """
        Compute distillation loss using the MultiStageDistiller's distillers directly.
        This is a compatibility wrapper for the trainer interface.
        """
        try:
            # Use the KDHintonDistiller directly since it's the only one configured
            if self.distiller.distillers:
                kd_distiller = self.distiller.distillers[0]  # First distiller should be KD
                if hasattr(kd_distiller, 'compute_loss'):
                    return kd_distiller.compute_loss(
                        student_logits=student_outputs.logits,
                        teacher_logits=teacher_outputs.logits,
                        labels=targets
                    )
            # Fallback to simple MSE loss between logits
            if hasattr(student_outputs, 'logits') and hasattr(teacher_outputs, 'logits'):
                return torch.nn.functional.mse_loss(student_outputs.logits, teacher_outputs.logits)
            else:
                return torch.tensor(0.0, device=self.device, requires_grad=True)
        except Exception as e:
            print(f"[WARNING] Distillation loss computation failed: {e}")
            # Fallback to simple MSE loss between logits
            if hasattr(student_outputs, 'logits') and hasattr(teacher_outputs, 'logits'):
                return torch.nn.functional.mse_loss(student_outputs.logits, teacher_outputs.logits)
            else:
                return torch.tensor(0.0, device=self.device, requires_grad=True)

    def _summarize_training(self, metrics_history):
        print('\n[SUMMARY] Training Summary:')
        for record in metrics_history:
            if not isinstance(record, dict):
                continue
            epoch = record.get('epoch', '?')
            train_loss = record.get('train_loss', 0)
            val_loss = record.get('val_loss', 0)
            accuracy = record.get('accuracy', 0)
            f1 = record.get('f1', 0)
            precision = record.get('precision', 0)
            recall = record.get('recall', 0)
            print(f'[SUMMARY] Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, '
                  f'Accuracy={accuracy:.4f}, F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}')
        # No plotting here to avoid repeated visualizations
