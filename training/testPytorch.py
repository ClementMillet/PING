
# Imports
import torch
from sklearn.metrics import confusion_matrix
import torch.nn as nn  
import torch.optim as optim  
import torchvision.transforms as transforms  
import torchvision
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import pandas as pd
from skimage import io
from torch.utils.data import (
    Dataset,
    DataLoader,
)  
import numpy as np


class CatsAndDogsDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = io.imread(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))

        if self.transform:
            image = self.transform(image)

        return (image, y_label)


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
in_channel = 3
num_classes = 2
learning_rate = 1e-3
batch_size = 4
num_epochs = 10

# Load Data
dataset = CatsAndDogsDataset(
    csv_file="rocks_no_rocks.csv",
    root_dir="rocks_no_rocks",
    transform=transforms.ToTensor(),
)


train_set, test_set = torch.utils.data.random_split(dataset, [953, 1192-953-1])
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

# Model
model = torchvision.models.googlenet(pretrained=True)
model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train Network
for epoch in range(num_epochs):
    losses = []

    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        losses.append(loss.item())

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()

    print(f"Cost at epoch {epoch} is {sum(losses)/len(losses)}")

# Check the confusion matrix and f-score to see the classifier efficiency 

def get_confusion_matrix(loader, model):
    target = np.array([])
    pred = np.array([])
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            scores = model(x)
            _, predictions = scores.max(1)
            target = np.concatenate((target,y.numpy()))
            pred = np.concatenate((pred,predictions.cpu().numpy()))
    model.train()
    return confusion_matrix(target, pred)

print('Finished Training')

cmt = get_confusion_matrix(test_loader, model)
print("confusion matrix : ")
print(cmt)
print("Correct predictions : ")
print(str(cmt.trace()) + "/" + str(cmt.sum()))
print("F-score : ") 
print(cmt[1,1]/(cmt[1,1]+1/2*(cmt[0,1] + cmt[1,0])))

PATH = './model.csv'
torch.save(model.state_dict(), PATH)
PATH = '../app/model.pth'
torch.save(model.state_dict(), PATH)
print('Finished Training')