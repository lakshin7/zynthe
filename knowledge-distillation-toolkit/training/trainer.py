import copy
from evaluation.metrics import compute_all_metrics
from evaluation.visualizer import plot_training_curves
from evaluation.metrics import plot_metrics
from evaluation.metrics_extended import (
    compute_extended_metrics,
    DistillationEfficacyIndex,
    CompressionAwareScore,
    LossComponentTracker
)
from core.distillers.multi_stage_distiller import DistillerRegistry
import torch
from torch.optim import AdamW
import os
import inspect
import json
import time

class Trainer:
    def __init__(self, teacher, student, tokenizer, config, device, experiment_dir):
        self.teacher = teacher
        self.student = student
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        self.experiment_dir = experiment_dir
        self.optimizer = AdamW(self.student.parameters(), lr=self.config['train'].get('lr', 5e-5))
        
        # Teacher training configuration
        self.should_train_teacher = self.config['train'].get('train_teacher', False)
        self.teacher_epochs = self.config['train'].get('teacher_epochs', 2)
        if self.should_train_teacher:
            self.teacher_optimizer = AdamW(self.teacher.parameters(), lr=self.config['train'].get('teacher_lr', 2e-5))
        
        # Initialize distiller from registry
        distil_cfg = self.config['distillation']
        registry = DistillerRegistry()
        distiller_type = distil_cfg.get('type', 'kd')
        
        # Get distiller class and instantiate with proper parameters
        distiller_class = registry.get(distiller_type)
        distiller_config = distil_cfg.get('config', {})
        
        self.distiller = distiller_class(
            teacher=self.teacher,
            student=self.student,
            device=self.device,
            **distiller_config
        )
        
        self.train_losses = []
        self.val_losses = []
        self.metrics_history = {'accuracy': [], 'f1': [], 'precision': [], 'recall': []}
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.early_stop_patience = self.config['train'].get('early_stop_patience', 2)
        self.no_improve_epochs = 0
        self.last_preds = []
        self.last_labels = []
        
        # Extended metrics tracking
        self.loss_tracker = LossComponentTracker()
        self.extended_metrics_history = {
            'kl_divergence': [],
            'js_divergence': [],
            'prediction_agreement': [],
            'confidence_correlation': []
        }
        
        # Cache model forward signatures for efficient parameter filtering
        self._teacher_forward_params = self._get_forward_params(self.teacher)
        self._student_forward_params = self._get_forward_params(self.student)

    def _get_forward_params(self, model):
        """Get the parameter names accepted by the model's forward method."""
        try:
            forward_signature = inspect.signature(model.forward)
            return set(forward_signature.parameters.keys())
        except Exception:
            # Fallback: assume standard transformer parameters
            return {'input_ids', 'attention_mask', 'labels'}
    
    def _filter_batch_for_model(self, batch, model_params):
        """Filter batch to only include parameters accepted by the model."""
        filtered_batch = {}
        for key, value in batch.items():
            if key in model_params:
                filtered_batch[key] = value
            else:
                # Log first few times to help debug, then suppress
                if not hasattr(self, '_param_warnings'):
                    self._param_warnings = set()
                if key not in self._param_warnings:
                    print(f"[DEBUG] Parameter '{key}' not supported by model, filtering out")
                    self._param_warnings.add(key)
        return filtered_batch

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
            
            # Filter batch for each model's specific requirements
            teacher_batch = self._filter_batch_for_model(batch, self._teacher_forward_params)
            student_batch = self._filter_batch_for_model(batch, self._student_forward_params)
            
            with torch.no_grad():
                try:
                    teacher_outputs = self.teacher(**teacher_batch)
                except Exception as e:
                    print(f"[WARNING] Teacher forward pass failed at batch {batch_idx}: {e}")
                    continue
            try:
                student_outputs = self.student(**student_batch)
            except Exception as e:
                print(f"[WARNING] Student forward pass failed at batch {batch_idx}: {e}")
                continue
            labels = batch.get('labels', None)
            try:
                result = self.distiller.compute_loss(
                    student_outputs=student_outputs,
                    teacher_outputs=teacher_outputs,
                    targets=labels
                )
                
                # Handle tuple return (loss, metrics_dict)
                if isinstance(result, tuple):
                    loss, metrics_dict = result
                else:
                    # Fallback if distiller returns only loss
                    loss = result
                    metrics_dict = {}
                    
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

    def evaluate(self, dataloader, compute_extended=True):
        self.student.eval()
        self.teacher.eval()
        total_loss = 0.0
        num_batches = 0
        all_preds = []
        all_labels = []
        
        # For extended metrics
        all_teacher_logits = []
        all_student_logits = []
        
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
                
                # Filter batch for each model's specific requirements
                teacher_batch = self._filter_batch_for_model(batch, self._teacher_forward_params)
                student_batch = self._filter_batch_for_model(batch, self._student_forward_params)
                
                try:
                    teacher_outputs = self.teacher(**teacher_batch)
                except Exception as e:
                    print(f"[WARNING] Teacher forward pass failed at batch {batch_idx} during evaluation: {e}")
                    continue
                try:
                    student_outputs = self.student(**student_batch)
                except Exception as e:
                    print(f"[WARNING] Student forward pass failed at batch {batch_idx} during evaluation: {e}")
                    continue
                labels = batch.get('labels', None)
                try:
                    result = self.distiller.compute_loss(
                        student_outputs=student_outputs,
                        teacher_outputs=teacher_outputs,
                        targets=labels
                    )
                    
                    # Handle tuple return (loss, metrics_dict)
                    if isinstance(result, tuple):
                        loss, metrics_dict = result
                    else:
                        # Fallback if distiller returns only loss
                        loss = result
                        metrics_dict = {}
                        
                except Exception as e:
                    print(f"[WARNING] Loss computation failed at batch {batch_idx} during evaluation: {e}")
                    loss = torch.tensor(0.0, device=self.device)
                    
                total_loss += loss.item()
                num_batches += 1
                
                # Extract predictions - handle dict, object, or tensor
                if labels is not None:
                    # Extract logits for both teacher and student
                    teacher_logits = teacher_outputs.logits if hasattr(teacher_outputs, 'logits') else teacher_outputs.get('logits') if isinstance(teacher_outputs, dict) else teacher_outputs
                    student_logits = student_outputs.logits if hasattr(student_outputs, 'logits') else student_outputs.get('logits') if isinstance(student_outputs, dict) else student_outputs
                    
                    # Store for extended metrics
                    if compute_extended and teacher_logits is not None and student_logits is not None:
                        all_teacher_logits.append(teacher_logits.cpu())
                        all_student_logits.append(student_logits.cpu())
                    
                    if student_logits is not None and hasattr(student_logits, 'dim') and student_logits.dim() >= 2:
                        preds = torch.argmax(student_logits, dim=-1)
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
        
        # Compute extended metrics if enabled
        extended_metrics = {}
        if compute_extended and all_teacher_logits and all_student_logits:
            try:
                teacher_logits_cat = torch.cat(all_teacher_logits, dim=0)
                student_logits_cat = torch.cat(all_student_logits, dim=0)
                
                temperature = self.config['distillation'].get('temperature', 2.0)
                extended_metrics = compute_extended_metrics(
                    teacher_logits_cat, 
                    student_logits_cat,
                    temperature=temperature
                )
                
                # Track extended metrics
                for key in self.extended_metrics_history.keys():
                    if key in extended_metrics:
                        self.extended_metrics_history[key].append(extended_metrics[key])
                
                print(f"[EXTENDED] KL: {extended_metrics['kl_divergence']:.4f}, "
                      f"Agreement: {extended_metrics['prediction_agreement']:.2%}")
            except Exception as e:
                print(f"[WARNING] Extended metrics computation failed: {e}")
        
        print(f"[EVAL] Evaluation completed. Average Loss: {avg_loss:.4f}, Metrics: {metrics if metrics else 'N/A'}")
        return avg_loss, metrics, extended_metrics

    def fit(self, train_loader, val_loader):
        # Optional: Train teacher first if configured
        if self.should_train_teacher:
            print(f"\n{'='*70}")
            print(f"PHASE 2.5: Fine-tuning Teacher Model")
            print(f"{'='*70}\n")
            print(f"[INFO] Training teacher for {self.teacher_epochs} epochs before distillation...")
            self._train_teacher(train_loader, val_loader)
            print(f"[INFO] Teacher training completed. Starting distillation...\n")
        
        metrics_history = []
        epochs = self.config['train']['epochs']
        print(f"[INFO] Training started for {epochs} epochs.")
        for epoch in range(epochs):
            print(f"[INFO] Starting epoch {epoch+1}/{epochs}")
            train_loss = self.train_epoch(train_loader)
            val_loss, val_metrics, extended = self.evaluate(val_loader, compute_extended=True)
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
            
            # Save student model
            student_save_dir = os.path.join(self.experiment_dir, 'student_model')
            self.student.save_pretrained(student_save_dir)
            self.tokenizer.save_pretrained(student_save_dir)
            print(f'[INFO] Student model and tokenizer saved to {student_save_dir}')
            
            # Save teacher model for comparison
            teacher_save_dir = os.path.join(self.experiment_dir, 'teacher_model')
            self.teacher.save_pretrained(teacher_save_dir)
            self.tokenizer.save_pretrained(teacher_save_dir)
            print(f'[INFO] Teacher model and tokenizer saved to {teacher_save_dir}')
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
        
        # Save extended metrics history
        try:
            extended_metrics_path = os.path.join(self.experiment_dir, 'extended_metrics.json')
            with open(extended_metrics_path, 'w') as f:
                json.dump(self.extended_metrics_history, f, indent=2)
            print(f"[INFO] Extended metrics saved to {extended_metrics_path}")
        except Exception as e:
            print(f"[WARNING] Failed to save extended metrics: {e}")

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

    def _train_teacher(self, train_loader, val_loader):
        """Train/fine-tune the teacher model on the task before distillation."""
        self.teacher.train()
        best_teacher_loss = float('inf')
        best_teacher_state = None
        
        for epoch in range(self.teacher_epochs):
            print(f"[TEACHER] Epoch {epoch+1}/{self.teacher_epochs}")
            
            # Training
            total_loss = 0.0
            num_batches = 0
            for batch_idx, batch in enumerate(train_loader):
                if not batch or not isinstance(batch, dict):
                    continue
                
                batch = {k: v.to(self.device) for k, v in batch.items() if hasattr(v, 'to')}
                teacher_batch = self._filter_batch_for_model(batch, self._teacher_forward_params)
                
                outputs = self.teacher(**teacher_batch)
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs['loss']
                
                self.teacher_optimizer.zero_grad()
                loss.backward()
                self.teacher_optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_train_loss = total_loss / max(num_batches, 1)
            
            # Validation
            self.teacher.eval()
            val_loss = 0.0
            val_batches = 0
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for batch in val_loader:
                    if not batch or not isinstance(batch, dict):
                        continue
                    
                    batch = {k: v.to(self.device) for k, v in batch.items() if hasattr(v, 'to')}
                    teacher_batch = self._filter_batch_for_model(batch, self._teacher_forward_params)
                    
                    outputs = self.teacher(**teacher_batch)
                    loss = outputs.loss if hasattr(outputs, 'loss') else outputs['loss']
                    val_loss += loss.item()
                    val_batches += 1
                    
                    # Get predictions
                    logits = outputs.logits if hasattr(outputs, 'logits') else outputs['logits']
                    preds = torch.argmax(logits, dim=-1)
                    labels = batch.get('labels')
                    
                    if labels is not None:
                        all_preds.extend(preds.cpu().numpy().tolist())
                        all_labels.extend(labels.cpu().numpy().tolist())
            
            avg_val_loss = val_loss / max(val_batches, 1)
            
            # Compute metrics
            teacher_metrics = {}
            if all_preds and all_labels:
                teacher_metrics = compute_all_metrics(all_preds, all_labels)
            
            accuracy = teacher_metrics.get('accuracy', 0)
            f1 = teacher_metrics.get('f1', 0)
            
            print(f"[TEACHER] Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, "
                  f"Val Loss={avg_val_loss:.4f}, Accuracy={accuracy:.4f}, F1={f1:.4f}")
            
            # Save best teacher
            if avg_val_loss < best_teacher_loss:
                best_teacher_loss = avg_val_loss
                best_teacher_state = copy.deepcopy(self.teacher.state_dict())
                print(f"[TEACHER] Best model updated (val_loss={avg_val_loss:.4f})")
            
            self.teacher.train()
        
        # Restore best teacher
        if best_teacher_state:
            self.teacher.load_state_dict(best_teacher_state)
            print(f"[TEACHER] Restored best teacher model (val_loss={best_teacher_loss:.4f})")
        
        # Set teacher back to eval mode for distillation
        self.teacher.eval()
