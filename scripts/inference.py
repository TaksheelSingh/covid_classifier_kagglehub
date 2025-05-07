
import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

# Paths
val_dir = r'C:\Users\TAKSHEEL\Downloads\covid_classifier_kagglehub-main\covid_classifier_kagglehub-main\datasets\COVID-19_Radiography_Dataset\val'
model_path = 'covid_classifier.pth'
batch_size = 16

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Dataset
val_dataset = datasets.ImageFolder(val_dir, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Print class index mapping for reference
print("🧠 Class-to-index mapping:", val_dataset.class_to_idx)

# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 3)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Predictions
y_true = []
y_pred = []
y_prob = []

print("🔁 Running inference with progress bar...")

with torch.no_grad():
    for images, labels in tqdm(val_loader, desc="Processing"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        predictions = torch.argmax(probs, dim=1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predictions.cpu().numpy())
        y_prob.extend(probs.cpu().numpy())  # ✅ Save full 3-class probabilities

# Save files
np.save("y_true.npy", np.array(y_true))
np.save("y_pred.npy", np.array(y_pred))
np.save("y_prob.npy", np.array(y_prob))

print("✅ Inference complete. Saved y_true.npy, y_pred.npy, y_prob.npy")
