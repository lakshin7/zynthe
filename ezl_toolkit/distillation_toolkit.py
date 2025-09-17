import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class DistillationTrainer:
    def __init__(self, teacher, student, device="auto"):
        # Device detection (works on any hardware)
        if device == "auto":
            self.device = torch.device(
                "mps" if torch.backends.mps.is_available() else 
                "cuda" if torch.cuda.is_available() else 
                "cpu"
            )
        else:
            self.device = torch.device(device)
            
        print(f"Using device: {self.device}")
        
        # Models
        self.teacher = teacher.to(self.device).eval()  # Teacher in eval mode
        self.student = student.to(self.device)
        
    def distill(self, 
                train_loader, 
                epochs=10, 
                temperature=4.0,
                alpha=0.5,  # CE vs KL loss balance
                student_only=False):
        """
        Performs knowledge distillation
        """
        optimizer = optim.Adam(self.student.parameters(), lr=1e-3)
        ce_loss = nn.CrossEntropyLoss()
        kl_loss = nn.KLDivLoss(reduction="batchmean")
        
        for epoch in range(epochs):
            self.student.train()
            running_loss = 0.0
            
            for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Forward pass
                with torch.no_grad():
                    teacher_logits = self.teacher(inputs)
                
                student_logits = self.student(inputs)
                
                # Calculate losses
                if student_only:
                    # Train student without teacher
                    loss = ce_loss(student_logits, labels)
                else:
                    # Standard distillation loss
                    soft_labels = nn.functional.softmax(teacher_logits / temperature, dim=-1)
                    soft_preds = nn.functional.log_softmax(student_logits / temperature, dim=-1)
                    
                    loss_kl = kl_loss(soft_preds, soft_labels) * (temperature**2)
                    loss_ce = ce_loss(student_logits, labels)
                    loss = alpha * loss_ce + (1 - alpha) * loss_kl
                
                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            print(f"Epoch {epoch+1} Loss: {running_loss/len(train_loader):.4f}")
        
        return self.student
    
    def evaluate(self, test_loader):
        """
        Compares teacher and student accuracy
        """
        def _accuracy(model, loader):
            correct = 0
            total = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            return correct / total
        
        teacher_acc = _accuracy(self.teacher, test_loader)
        student_acc = _accuracy(self.student, test_loader)
        
        print("\n" + "="*50)
        print(f"Teacher Accuracy: {teacher_acc*100:.2f}%")
        print(f"Student Accuracy: {student_acc*100:.2f}%")
        print("="*50)
        
        return {"teacher": teacher_acc, "student": student_acc}