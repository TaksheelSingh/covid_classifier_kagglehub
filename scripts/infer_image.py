import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import sys
import os

# ✅ 1. Validate input
if len(sys.argv) != 2:
    print("Usage: python infer_image.py <image_path>")
    sys.exit(1)

image_path = sys.argv[1]
if not os.path.exists(image_path):
    print(f"❌ Error: File not found - {image_path}")
    sys.exit(1)

# ✅ 2. Define class labels
classes = ["COVID-19", "Normal", "Viral Pneumonia"]

# ✅ 3. Load the trained model
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, len(classes))
model.load_state_dict(torch.load("covid_classifier.pth", map_location=torch.device("cpu")))
model.eval()

# ✅ 4. Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Ensure this matches training
])

# ✅ 5. Load image
try:
    image = Image.open(image_path).convert("RGB")
except Exception as e:
    print(f"❌ Failed to load image: {e}")
    sys.exit(1)

image = transform(image).unsqueeze(0)  # Add batch dimension

# ✅ 6. Inference
with torch.no_grad():
    output = model(image)
    _, predicted = torch.max(output, 1)

# ✅ 7. Output result
print(f"🔍 Prediction: {classes[predicted.item()]}")
