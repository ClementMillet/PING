from torchutils import transform_image, get_prediction
import torch
from torchvision import models
from PIL import Image
import torch.nn as nn
import csv
import copy

im_dir = "/mnt/d/Users/cleme/Pictures/jackolantern.jpg"

PATH = "model.pth"

model = models.resnet18(pretrained=True)
model_ori = copy.deepcopy(model)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
model.eval()
model_ori.eval()

input_size = 224

imagenet_class = []

with open('imagenet_class.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        imagenet_class.append(row)

im = Image.open(im_dir)
tensor = transform_image(im, input_size)
prediction = get_prediction(tensor, model, model_ori, imagenet_class)

print(prediction)
