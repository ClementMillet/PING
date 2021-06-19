import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import io

PATH = "model.pth"

input_size = 224

model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
model.eval()

def transform_image(image):
    transform=transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def get_prediction(image_tensor):
    outputs = model(image_tensor)
        # max returns (value ,index)
    #_, predicted = torch.max(outputs.data, 1)
    return outputs.data
