import torch
from torchvision import models
import torchvision.transforms as transforms
import numpy as np
import csv
from PIL import Image
import io
import copy

def transform_image(image, input_size):
    image = image.convert('RGB')
    transform=transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def get_prediction(image_tensor, model, model_ori, imagenet_class):
    output = model(image_tensor)
    _, predicted = torch.max(output.data, 1)
    if predicted == 1:
        return ['rock', torch.nn.functional.softmax(output.data)]
    else:
        outputs = model_ori(image_tensor)
        proba, predicted = torch.topk(torch.nn.functional.softmax(outputs.data), 3)
        return [[imagenet_class[k][0] for k in predicted[0]], proba[0]]
