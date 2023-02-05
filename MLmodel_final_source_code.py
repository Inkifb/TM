import os
import torch
import numpy
import pandas as pd
import torchvision
import torchvision.models as models
import matplotlib.pyplot as plt
import requests
from PIL import Image
from pathlib import Path
from torchvision.utils import make_grid
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data.dataloader import DataLoader
batch_size = 32
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics

confusion_matrix = [[0,0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0,0]]

full_labels = []
full_preds = []

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
data_dir = '/home/kpaksaran/ondemand/dataset/dataset_huh'
classes = os.listdir(data_dir)
print(classes)
transformer = transforms.Compose([transforms.Resize((224, 256)), transforms.ColorJitter(brightness=0.25), transforms.ToTensor()])
dataset = ImageFolder(data_dir, transform = transformer)

random_seed = 42
torch.manual_seed(random_seed)
train_ds, val_ds, test_ds = random_split(dataset, [2995, 749, 29])
len(train_ds), len(val_ds), len(test_ds)

train_dl = DataLoader(train_ds, batch_size, shuffle = True, num_workers = 4, pin_memory = True)
val_dl = DataLoader(val_ds, batch_size*2, num_workers = 4, pin_memory = True)

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    label_list = labels.tolist()
    preds_list = preds.tolist()

    full_labels.extend(label_list)
    full_preds.extend(preds_list)
    for i in range(len(label_list)):
      confusion_matrix[preds_list[i]][label_list[i]] = confusion_matrix[preds_list[i]][label_list[i]] + 1

    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class ImageClassificationBase(nn.Module):

    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # generate predictions -training
        loss = F.cross_entropy(out, labels) # calculate loss (gradient) - criterion
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # generate predictions - validation
        loss = F.cross_entropy(out, labels)   # calculate loss (gradient) - criterion
        acc = accuracy(out, labels)           # calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        # TO-DO: add time per epoch
        print("Epoch {}: train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch+1, result['train_loss'], result['val_loss'], result['val_acc']))
    
class ResNet50(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1) # use a pretrained model
        num_ftrs = self.network.fc.in_features # collect number of features
        self.network.fc = nn.Linear(num_ftrs, len(dataset.classes)) # replace last layer

        self.name = 'ResNet50'
    
    def forward(self, xb):
        return torch.sigmoid(self.network(xb))

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)
    
class DeviceDataLoader():
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        return len(self.dl)


model = ResNet50()
model.to(device)

train_dl = DeviceDataLoader(train_dl, device)
val_dl = DeviceDataLoader(val_dl, device)

@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase 
        model.train()
        train_losses = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
    return history

num_epochs = 20
opt_func = torch.optim.Adam
lr = 3e-6
history = fit(num_epochs, lr, model, train_dl, val_dl, opt_func)
print(history)

for i in confusion_matrix:
  print(i)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = metrics.confusion_matrix(full_labels, full_preds), display_labels = [0,1,2,3,4,5,6,7,8,9,10])
cm_display.plot()

print("Recall: " + str(round(metrics.recall_score(full_labels, full_preds, average="macro") * 100) / 100))

print("Precision: " + str(round(metrics.precision_score(full_labels, full_preds, average="macro") * 100) / 100))

print("F1_score: " + str(round(metrics.f1_score(full_labels, full_preds, average="macro") * 100) / 100))

plt.show()

torch.save(model, '/home/kpaksaran/ondemand/dataset/Version_aftertrain/Version14.1_ResNet.pth')
