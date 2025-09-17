# File: demo.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# ========================
# 1. Distillation Trainer
# ========================
class DistillationTrainer:
    def __init__(self, teacher, student, device="auto"):
        # Auto-detect best device for Apple Silicon
        if device == "auto":
            self.device = torch.device(
                "mps" if torch.backends.mps.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)
        
        print(f"🍏 Using Apple MPS: {self.device.type.upper()}" if self.device.type == 'mps' 
              else f"⚙️ Using device: {self.device}")
        
        self.teacher = teacher.to(self.device).eval()
        self.student = student.to(self.device)
    
    def distill(self, train_loader, epochs=10, temperature=4.0, alpha=0.5, student_only=False):
        optimizer = optim.Adam(self.student.parameters(), lr=0.001)
        ce_loss = nn.CrossEntropyLoss()
        kl_loss = nn.KLDivLoss(reduction="batchmean")
        
        # Training loop with MPS optimizations
        for epoch in range(epochs):
            self.student.train()
            total_loss = 0
            
            for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                with torch.no_grad():
                    teacher_logits = self.teacher(inputs)
                
                student_logits = self.student(inputs)
                
                if student_only:
                    loss = ce_loss(student_logits, labels)
                else:
                    # Use torch.no_grad() for teacher outputs
                    with torch.no_grad():
                        soft_labels = F.softmax(teacher_logits / temperature, dim=-1)
                    
                    soft_preds = F.log_softmax(student_logits / temperature, dim=-1)
                    loss_kl = kl_loss(soft_preds, soft_labels) * (temperature ** 2)
                    loss_ce = ce_loss(student_logits, labels)
                    loss = alpha * loss_ce + (1 - alpha) * loss_kl
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            print(f"📉 Loss: {avg_loss:.4f}")
            
            # Clear MPS cache to prevent memory buildup
            if self.device.type == 'mps':
                torch.mps.empty_cache()
        
        return self.student
    
    def evaluate(self, test_loader):
        def _accuracy(model, loader):
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (preds == labels).sum().item()
            return correct / total
        
        teacher_acc = _accuracy(self.teacher, test_loader)
        student_acc = _accuracy(self.student, test_loader)
        
        print("\n" + "="*50)
        print(f"👨🏫 Teacher Accuracy: {teacher_acc*100:.2f}%")
        print(f"👨🎓 Student Accuracy: {student_acc*100:.2f}%")
        print("="*50)
        
        return {"teacher": teacher_acc, "student": student_acc}

# ========================
# 2. Model Architectures
# ========================
class CIFARTeacher(nn.Module):
    """Teacher model (~4.2M params)"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.3)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class CIFARStudent(nn.Module):
    """Student model (~0.3M params)"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc = nn.Linear(64 * 8 * 8, 10)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# ========================
# 3. Main Execution
# ========================
if __name__ == "__main__":
    # Auto-detect device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # 1. Load CIFAR-10 dataset
    print("Loading CIFAR-10 dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_set = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, pin_memory=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=128, pin_memory=True, num_workers=2)
    
    # 2. Initialize models
    print("\nInitializing models...")
    teacher = CIFARTeacher()
    student = CIFARStudent()
    
    # 3. Pre-train teacher first
    print("\n🔥 Pre-training teacher model...")
    teacher = teacher.to(device)
    teacher_optim = optim.Adam(teacher.parameters(), lr=0.001)
    ce_loss = nn.CrossEntropyLoss()
    
    for epoch in range(10):
        teacher.train()
        total_loss = 0
        for inputs, labels in tqdm(train_loader, desc=f"Teacher Epoch {epoch+1}/10"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            teacher_optim.zero_grad()
            outputs = teacher(inputs)
            loss = ce_loss(outputs, labels)
            loss.backward()
            teacher_optim.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"📉 Teacher Loss: {avg_loss:.4f}")
        
        # Clear MPS cache
        if device.type == 'mps':
            torch.mps.empty_cache()
    
    # 4. Evaluate teacher
    teacher.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = teacher(inputs)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    
    teacher_acc = correct / total
    print(f"\n✅ Teacher Accuracy: {teacher_acc*100:.2f}%")
    
    # 5. Initialize distillation trainer
    trainer = DistillationTrainer(teacher, student)
    
    # 6. Train student without distillation (baseline)
    print("\n🚀 Training student without distillation...")
    trainer.distill(train_loader, epochs=10, student_only=True)
    baseline_results = trainer.evaluate(test_loader)
    
    # 7. Perform knowledge distillation
    print("\n🧠 Distilling knowledge to student...")
    trainer.distill(train_loader, epochs=10, temperature=3.0, alpha=0.7)
    distilled_results = trainer.evaluate(test_loader)
    
    # 8. Save student model
    torch.save(trainer.student.state_dict(), "distilled_student.pth")
    
    # 9. Visualize results
    plt.figure(figsize=(10, 6))
    plt.bar(['Teacher', 'Student (Raw)', 'Student (Distilled)'], 
            [teacher_acc * 100, baseline_results['student'] * 100, distilled_results['student'] * 100],
            color=['blue', 'orange', 'green'])
    
    plt.ylabel('Accuracy (%)')
    plt.title('Knowledge Distillation Results on CIFAR-10')
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels
    for i, v in enumerate([teacher_acc * 100, baseline_results['student'] * 100, distilled_results['student'] * 100]):
        plt.text(i, v + 1, f"{v:.1f}%", ha='center')
    
    plt.savefig('distillation_results.png', bbox_inches='tight')
    plt.show()
    
    print("\n✅ Distillation complete! Student model saved as 'distilled_student.pth'")