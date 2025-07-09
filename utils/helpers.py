import torch
import cv2
from torchvision import transforms, models
import numpy as np

def load_model(model_path):
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model = models.resnet50(pretrained=False)
    num_features = model.fc.in_features
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(num_features, 512),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(512, len(checkpoint['class_names']))
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint['class_names']

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def predict(image, model, class_names):
    with torch.no_grad():
        outputs = model(image)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
        confidence, pred_idx = torch.max(probs, 0)
    return class_names[pred_idx.item()], confidence.item()