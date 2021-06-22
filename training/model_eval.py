#%%
import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
import numpy as np
from os import listdir
from PIL import Image
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
try:
    import matplotlib
    import matplotlib.pyplot as plt
except: 
    print("No matplotlib module found")


#--Variables--

target = np.array([])
prediction = np.array([])

PATH = '/mnt/d/Users/cleme/Pictures/PING/rocks_no_rocks/test/'

model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model.load_state_dict(torch.load('../app/model.pth'))
model.eval()

input_size = 224

#--Results--

def transform_image(image):
    transform=transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def get_prediction(tensor):
    outputs = model(tensor)
    _, predicted = torch.max(outputs.data, 1)
    return predicted

for f in listdir(PATH + 'rock'):
    im = Image.open(PATH + 'rock/' + f)
    tensor = transform_image(im)
    print(f)
    pred = get_prediction(tensor)
    target = np.concatenate((target, [1]))
    prediction = np.concatenate((prediction, pred))

for f in listdir(PATH + 'no_rock'):
    im = Image.open(PATH + 'no_rock/' + f)
    tensor = transform_image(im)
    print(f)
    pred = get_prediction(tensor)
    target = np.concatenate((target, [0]))
    prediction = np.concatenate((prediction, pred))

print()

print('Confusion matrix :')
cmt = confusion_matrix(target, prediction)
print(cmt)
print()
f_score = f1_score(target, prediction, zero_division = 1)
print('F-score : {:.4f}'.format(f_score))
print()

try:
    fig, ax = plt.subplots()
    im = ax.imshow(cmt)

    labels = ['no rock', 'rock']

    ax.set_xlabel('Prediction')
    ax.set_ylabel('Target')

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))

    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, cmt[i, j], ha="center", va="center", color="r")

    ax.set_title("Confusion Matrix")

    fig.tight_layout()
    plt.show()
    
except:
    print("No matplotlib module found")