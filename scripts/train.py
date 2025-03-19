import os
import shutil
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# ✅ Set dataset path (UPDATE this if needed)
dataset_path = r"C:\Users\TAKSHEEL\Downloads\covid_classifier_kagglehub\datasets\COVID-19_Radiography_Dataset"
train_path = os.path.join(dataset_path, "train")
val_path = os.path.join(dataset_path, "val")

# ✅ Check if dataset exists
if not os.path.exists(dataset_path):
    print(f"❌ Dataset path not found: {dataset_path}")
    exit()

# ✅ Create train and validation folders if not exist
for category in ["COVID", "Normal", "Viral Pneumonia"]:
    os.makedirs(os.path.join(train_path, category), exist_ok=True)
    os.makedirs(os.path.join(val_path, category), exist_ok=True)

# ✅ Split dataset manually (80% Train, 20% Validation)
split_ratio = 0.8  # 80% training, 20% validation

for category in ["COVID", "Normal", "Viral Pneumonia"]:
    folder_path = os.path.join(dataset_path, category)
    
    if not os.path.exists(folder_path):
        print(f"❌ Folder not found: {folder_path}")
        continue

    images = os.listdir(folder_path)
    random.shuffle(images)

    train_size = int(len(images) * split_ratio)
    train_images = images[:train_size]
    val_images = images[train_size:]

    # Move images to train/val folders
    for img in train_images:
        shutil.move(os.path.join(folder_path, img), os.path.join(train_path, category, img))
    for img in val_images:
        shutil.move(os.path.join(folder_path, img), os.path.join(val_path, category, img))

print("✅ Dataset successfully split into Train/Validation sets.")

# ✅ Define image transformations
data_transforms = {
    "train": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ]),
    "val": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
}

# ✅ Load dataset using ImageFolder
datasets_dict = {
    "train": datasets.ImageFolder(train_path, transform=data_transforms["train"]),
    "val": datasets.ImageFolder(val_path, transform=data_transforms["val"])
}

# ✅ Create DataLoader
batch_size = 32
dataloaders = {
    "train": DataLoader(datasets_dict["train"], batch_size=batch_size, shuffle=True),
    "val": DataLoader(datasets_dict["val"], batch_size=batch_size, shuffle=False)
}

# ✅ Load Pretrained Model (ResNet18)
model = models.resnet18(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 3)  # 3 Classes: COVID-19, Normal, Viral Pneumonia

# ✅ Define Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ✅ Train the Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = 5

for epoch in range(num_epochs):
    print(f"🔹 Epoch {epoch+1}/{num_epochs} 🔹")

    for phase in ["train", "val"]:
        if phase == "train":
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in dataloaders[phase]:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == "train"):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase == "train":
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / len(dataloaders[phase])
        epoch_acc = correct / total

        print(f"✅ {phase.capitalize()} Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

print("🎉 Training Complete!")

# ✅ Save the trained model
torch.save(model.state_dict(), "covid_classifier.pth")
print("📁 Model saved as covid_classifier.pth")
