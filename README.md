# Small AI Project: Waste Classification
# by me: Ahmed Salem Albraiki

import torch
from torchvision import models, transforms
from PIL import Image
import os

# ----- Settings -----
IMG_PATH = "test_image.jpg"  # Replace with your image path
CLASSES = ["recyclable", "non_recyclable"]  # Adjust classes
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- Pre-trained Model -----
model = models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, len(CLASSES))
model = model.to(DEVICE)
model.eval()

# If you have a saved model, load it:
# model.load_state_dict(torch.load("smart_waste_model.pth"))

# ----- Image Transform -----
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

# ----- Prediction -----
img = Image.open(IMG_PATH).convert("RGB")
img_t = transform(img).unsqueeze(0).to(DEVICE)
with torch.no_grad():
    output = model(img_t)
    _, pred = torch.max(output, 1)
    print(f"Prediction: {CLASSES[pred.item()]}")
