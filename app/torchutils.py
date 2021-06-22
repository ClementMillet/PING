import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models
import torchvision.transforms as transforms
import numpy as np
import csv
from PIL import Image
import io
import copy

PATH = "model.pth"

input_size = 224

imagenet_class = []

with open('imagenet_class.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        imagenet_class.append(row)

model = models.resnet18(pretrained=True)
model_ori = copy.deepcopy(model)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
model.eval()
model_ori.eval()

def transform_image(image):
    image = image.convert('RGB')
    transform=transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def get_prediction(image_tensor):
    output = model(image_tensor)
    _, predicted = torch.max(output.data, 1)
    if predicted == 1:
        return ['rock', torch.max(output.data)]
    else:
        outputs = model_ori(image_tensor)
        _,predicted = torch.max(outputs.data, 1)
        return [imagenet_class[predicted][0], torch.max(outputs.data)]
