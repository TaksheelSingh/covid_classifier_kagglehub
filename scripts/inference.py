import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import sys

# Load the trained model
model = models.resnet18(weights=None)  # No pre-trained weights
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 3)  # 3 classes (COVID, Normal, Viral Pneumonia)
model.load_state_dict(torch.load("covid_classifier.pth", map_location=torch.device("cpu")))
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Get the image path from command line
image_path = sys.argv[1]  # Reads the input image path

# Load and preprocess the image
image = Image.open(image_path).convert("RGB")
image = transform(image).unsqueeze(0)  # Add batch dimension

# Perform inference
with torch.no_grad():
    output = model(image)
    _, predicted = torch.max(output, 1)

# Class labels
classes = ["COVID-19", "Normal", "Viral Pneumonia"]
print(f"🔍 Prediction: {classes[predicted.item()]}")
