import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import io

# Set device
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = torchvision.models.googlenet(pretrained=True)

PATH = "model.pth"
model.load_state_dict(torch.load(PATH,map_location=torch.device('cpu')))
model.eval()

def transform_image(image):
    transform=transforms.Compose([transforms.ToTensor()])
    return transform(image).unsqueeze(0)

def get_prediction(image_tensor):
    outputs = model(image_tensor)
        # max returns (value ,index)
    _, predicted = torch.max(outputs.data, 1)
    return predicted
