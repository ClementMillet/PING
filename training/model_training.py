from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
import time
import os
import copy
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#--Variables--

data_dir = "/mnt/d/Users/cleme/Pictures/PING/rocks_no_rocks" 
PATH = '../app/'
num_class = 2
batch_size = 4
num_epoch = 15
input_size = 224
feature_extract = False

#--Load Data--

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

print("Initializing Datasets and Dataloaders...")

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'test']}
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'test']}

#--Initialize Model--

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

model = models.resnet18(pretrained=True)
#torch.save(model.state_dict(), PATH + 'model_ori.pth')
num_ftrs = model.fc.in_features
set_parameter_requires_grad(model, feature_extract)
model.fc = nn.Linear(num_ftrs, num_class)

#--Training and Validation

model.to(device)

params_to_update = model.parameters()
if feature_extract:
    params_to_update = []
    for _,param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)

optimizer = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

criterion = nn.CrossEntropyLoss()

def train_model(model, dataloader, optimizer, criterion, num_epoch):
    since = time.time()

    accuracy_history = []

    best_model = copy.deepcopy(model.state_dict())
    best_accuracy = 0

    for epoch in range(num_epoch):
        print('Epoch {}/{}'.format(epoch, num_epoch - 1))
        print('-' * 10)

        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            run_loss = 0
            run_correct = 0

            for input, label in dataloader[phase]:
                input = input.to(device)
                label = label.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    output = model(input)
                    loss = criterion(output, label)
                    _, pred = torch.max(output, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                run_loss += loss.item() * input.size(0)
                run_correct += torch.sum(pred == label.data)
                
            epoch_loss = run_loss / len(dataloader[phase].dataset)
            epoch_accuracy = run_correct.double() / len(dataloader[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_accuracy))

            if phase == 'test' and epoch_accuracy > best_accuracy:
                best_accuracy = epoch_accuracy
                best_model = copy.deepcopy(model.state_dict())
            if phase == 'test':
                accuracy_history.append(epoch_accuracy)

        print()
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_accuracy))

    model.load_state_dict(best_model)
    return model, accuracy_history
                        
model, hist = train_model(model, dataloaders_dict, optimizer, criterion, num_epoch)

print()

torch.save(model.state_dict(), PATH + 'model.pth')
print("Model saved")
