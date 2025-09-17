# File: test_model.py
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np

# 1. Define your student model architecture (must match training)
class CIFARStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 10)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# 2. Load the trained model
def load_model(model_path):
    model = CIFARStudent()
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set to evaluation mode
    return model

# 3. Prepare test data
def prepare_test_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    test_set = datasets.CIFAR10(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    # Create a small subset for visualization (first 10 images)
    indices = torch.arange(10)
    test_subset = torch.utils.data.Subset(test_set, indices)
    
    return test_set, test_subset

# 4. Evaluate model accuracy
def evaluate_accuracy(model, test_loader):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f"📊 Test Accuracy: {accuracy:.2f}%")
    return accuracy

# 5. Visualize predictions
def visualize_predictions(model, test_subset):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    
    # CIFAR-10 class names
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck')
    
    # Create a grid of test images
    test_loader = torch.utils.data.DataLoader(test_subset, batch_size=10, shuffle=False)
    images, labels = next(iter(test_loader))
    
    # Get predictions
    with torch.no_grad():
        outputs = model(images.to(device))
        _, predicted = torch.max(outputs, 1)
    
    # Denormalize images for display
    def denormalize(img):
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
        std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)
        return img * std + mean
    
    # Create figure
    plt.figure(figsize=(15, 8))
    for i in range(10):
        plt.subplot(2, 5, i+1)
        
        # Denormalize and convert to numpy
        img = denormalize(images[i]).cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        img = np.clip(img, 0, 1)
        
        plt.imshow(img)
        plt.axis('off')
        
        # Set title color based on prediction correctness
        color = "green" if predicted[i] == labels[i] else "red"
        title = f"True: {classes[labels[i]]}\nPred: {classes[predicted[i]]}"
        plt.title(title, color=color, fontsize=10)
    
    plt.tight_layout()
    plt.savefig('model_predictions.png', dpi=120)
    plt.show()

# 6. Main function
def main():
    # Load the trained model
    model_path = "distilled_student.pth"
    model = load_model(model_path)
    print("✅ Model loaded successfully")
    
    # Prepare test data
    test_set, test_subset = prepare_test_data()
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False)
    
    # Evaluate accuracy
    accuracy = evaluate_accuracy(model, test_loader)
    
    # Visualize predictions
    visualize_predictions(model, test_subset)

if __name__ == "__main__":
    main()