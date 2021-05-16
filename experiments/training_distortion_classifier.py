import os, time
import torch
import torch.nn as nn
import numpy as np
import sys, time, math
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision
import torchvision.models as models
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.utils import save_image
import os, cv2
from torchvision import transforms, utils, datasets
from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler, WeightedRandomSampler
import copy
import matplotlib.pyplot as plt
from PIL import Image
from distortionNet import DistortionNet

class LoadDataset():
  def __init__(self, input_dim, batch_size_train, batch_size_test):
    self.input_dim = input_dim
    self.batch_size_train = batch_size_train
    self.batch_size_test = batch_size_test

    self.transformation_list = transforms.Compose([transforms.Resize(input_dim),
                                                   transforms.CenterCrop(input_dim),
                                                   transforms.ToTensor(),])


  def cifar_10(self):
    # Load Cifar-10 dataset 
    root = "cifar_10"

    trainset = datasets.CIFAR10(root=root, train=True, download=True,
                                transform=self.transformation_list)
    
    trainLoader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size_train, 
                                              num_workers=2, shuffle=True, drop_last=True)
    
    testset = datasets.CIFAR10(root=root, train=False, download=True,
                               transform=self.transformation_list)
    
    testLoader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size_test, num_workers=2, shuffle=False)
    
    return trainLoader, testLoader

  def cifar_100(self):
    # Load Cifar-100 dataset
    root = "cifar_100"

    trainset = datasets.CIFAR100(root=root, train=True, download=True,
                                transform=transforms.Compose(self.transformation_list))
    
    trainLoader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size_train, 
                                              num_workers=2, shuffle=True, drop_last=True)
    
    testset = datasets.CIFAR100(root=root, train=False, download=True,
                               transform=transforms.Compose(self.transformation_list))
    
    testLoader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size_test, num_workers=4, shuffle=False)
    
    return trainLoader, testLoader

  def imageNet(self, root_path):
    # Load ImageNet Dataset

    test_dataset = datasets.ImageFolder(root = root_path, transform = self.transformation_list)
    _, val_dataset = random_split(test_dataset, (20000, 30000))

    val_loader = DataLoader(dataset=val_dataset, shuffle=False, batch_size=self.batch_size_test)
    return None, val_loader

  def customDataset(self, root_path, split_train=0.8):
    dataset = datasets.ImageFolder(root = root_path, transform = self.transformation_list)
    train_size = int(split_train*len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, (train_size, test_size))
    train_dataset, val_dataset =  random_split(train_dataset, (int(split_train*len(train_dataset)), len(train_dataset) - int(split_train*len(train_dataset))))   
    
    train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=self.batch_size_train, num_workers=4)
    val_loader = DataLoader(dataset=val_dataset, shuffle=False, batch_size=self.batch_size_test, num_workers=4)
    test_loader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=self.batch_size_test, num_workers=4)
    return train_loader, val_loader, test_loader 



class EarlyStopping:
  def __init__(self, patience=10, verbose=False, delta=0, savePath="./model.pt"):
    self.patience = patience
    self.verbose = verbose
    self.delta = delta
    self.savePath = savePath
    self.best_loss = None
    self.count = 0
    self.shouldStop = False

  def __call__(self, val_loss, model, epoch, optimizer, val_acc):
    if (self.best_loss is None):
      self.save_checkpoint(val_loss, model, epoch, optimizer, val_acc)
    elif (val_loss > (self.best_loss-delta)):
      self.count += 1
      if (self.count > self.patience):
        self.shouldStop = True
    else:
      self.save_checkpoint(val_loss, model, epoch, optimizer, val_acc)
      if (self.verbose):
        print("Validation Loss has decreased from %s to %s"%(val_loss, self.best_loss))
      self.best_loss = val_loss
      self.count = 0

  def save_checkpoint(self, val_loss, model, epoch, optimizer, val_acc):
    torch.save({"epoch": epoch,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "model_state_dict": model.state_dict(),
                "optim_state_dict": optimizer.state_dict()}, self.savePath)

def train(model, trainLoader, optimizer, criterion, epoch, device):
  loss_list = []
  acc_list = []
  model.train()
  
  for i, (data, target) in enumerate(trainLoader):
    print("Batch: %s/%s"%(i, len(trainLoader)))
    data, target = data.to(device), target.to(device, dtype=torch.int64)
    optimizer.zero_grad()
    outputs = model(data)

    loss = criterion(outputs, target)
    loss.backward()
    optimizer.step()
    
    loss_list.append(loss.item())
    _, inf_label = torch.max(outputs, 1)
    acc = 100*(inf_label.eq(target.view_as(inf_label)).sum().item()/data.size(0))
    #acc = accuracy(outputs, target, topk=(1, ))
    acc_list.append(acc)
    #print("loss: %s"%(loss.item()))

  avg_acc, avg_loss = np.mean(acc_list), np.mean(loss_list)
  print("Epoch: %s, Avg Train Loss: %s, Avg Train Acc: %s"%(epoch, avg_acc, avg_loss))
  return avg_acc, avg_loss

def evaluate(model, valLoader, criterion, epoch, device):
  loss_list = []
  acc_list = []
  model.eval()
  for i, (data, target) in enumerate(valLoader):
    data, target = data.to(device), target.to(device, dtype=torch.int64)
    outputs = model(data)
    loss = criterion(outputs, target)
    loss_list.append(loss.item())
    _, inf_label = torch.max(outputs, 1)
    acc = 100*(inf_label.eq(target.view_as(inf_label)).sum().item()/data.size(0))
    acc_list.append(acc)

  avg_acc, avg_loss = np.mean(acc_list), np.mean(loss_list)
  print("Epoch: %s, Avg Val Loss: %s, Avg Val Acc: %s"%(epoch, avg_acc, avg_loss))
  return avg_acc, avg_loss



parser = argparse.ArgumentParser(description='Evaluating DNNs perfomance using distorted image: blur ou gaussian noise')
parser.add_argument('--distortion_type', type=str, default="pristine", 
  choices=['pristine', 'gaussian_blur','gaussian_noise'], help='Distortion Type (default: pristine)')
parser.add_argument('--root_path', type=str, help='Path to the distorted datasets')
parser.add_argument('--save_path', type=str, help='Path to save the distortion classifier model')

args = parser.parse_args()


input_dim = 224
batch_size_train, batch_size_test = 128, 128
root_path = args.root_path
dataset = LoadDataset(input_dim, batch_size_train, batch_size_test)
trainLoader, valLoader, testLoader = dataset.customDataset(root_path, split_train=0.8)
num_epochs = 50
learning_rate = 0.001
weight_decay = 0.0005
patience = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
saveModelPath = args.save_path
saveHistoryPath = "./distortion_classifier_history.pdf"
model = DistortionNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
criterion = nn.CrossEntropyLoss().to(device)
train_loss_list, val_loss_list = [], []
early_stopping = EarlyStopping(patience=patience, savePath=saveModelPath)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')

for epoch in range(1, num_epochs + 1):
  print("Epoch: %s"%(epoch))
  train_accuracy, train_loss = train(model, trainLoader, optimizer, criterion, epoch, device)
  val_accuracy, val_loss = evaluate(model, valLoader, criterion, epoch, device)
  scheduler.step(val_loss)
  train_loss_list.append(train_loss)
  val_loss_list.append(val_loss)
  early_stopping(val_loss, model, epoch, optimizer, val_accuracy)
  if (early_stopping.shouldStop):
    print("STOP!")
    break
