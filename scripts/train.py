import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, WeightedRandomSampler
from collections import Counter
import copy

# ✅ Paths
train_dir = r'datasets/COVID-19_Radiography_Dataset/train'
val_dir = r'datasets/COVID-19_Radiography_Dataset/val'

# ✅ Hyperparameters
batch_size = 32
num_epochs = 20
learning_rate = 0.001
patience = 3  # Early stopping patience

# ✅ Image transforms (ImageNet normalization)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ✅ Load datasets
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
val_dataset = datasets.ImageFolder(val_dir, transform=transform)

# ✅ Class-balanced sampling
targets = train_dataset.targets
class_counts = Counter(targets)
class_weights = [1.0 / class_counts[label] for label in targets]
sampler = WeightedRandomSampler(class_weights, len(class_weights), replacement=True)

# ✅ DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# ✅ Model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 3)
model = model.to(device)

# ✅ Loss & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ✅ Early stopping setup
best_val_loss = float("inf")
patience_counter = 0
best_model_weights = copy.deepcopy(model.state_dict())

# ✅ Training loop
for epoch in range(num_epochs):
    print(f"\n🔹 Epoch {epoch+1}/{num_epochs}")
    model.train()
    total, correct = 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    val_loss = 0.0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            total += labels.size(0)

    avg_val_loss = val_loss / total
    print(f"✅ Validation Loss: {avg_val_loss:.4f}")

    # Early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_weights = copy.deepcopy(model.state_dict())
        patience_counter = 0
        print("💾 Best model updated.")
    else:
        patience_counter += 1
        print(f"⏳ No improvement ({patience_counter}/{patience})")

    if patience_counter >= patience:
        print("🛑 Early stopping triggered.")
        break

# ✅ Save best model
model.load_state_dict(best_model_weights)
torch.save(model.state_dict(), "covid_classifier.pth")
print("📁 Model saved as covid_classifier.pth")
