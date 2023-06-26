import torch
import torch.nn as nn
import numpy as np
import sys, time, math, os
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from scipy.stats import entropy
import pandas as pd
import torchvision.transforms as transforms
import torchvision
import torchvision.models as models
from torchvision import datasets, transforms
from scipy import stats
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.datasets.voc as voc
from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler
from pthflops import count_ops
from mobileNet import B_MobileNet
import cv2
from PIL import Image
import argparse
from tqdm import tqdm

class AddGaussianNoise(object):
  def __init__(self, distortion_list, mean=0.):
    self.distortion_list = distortion_list

  def __call__(self, img):
    image = np.array(img)
    self.std = self.distortion_list[np.random.choice(len(self.distortion_list), 1)[0]]
    noise_img = image + np.random.normal(0, self.std, (image.shape[0], image.shape[1], image.shape[2]))
    return Image.fromarray(np.uint8(noise_img)) 
    
  def __repr__(self):
    return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class AddGaussianBlur(object):
	def __init__(self, distortion_list, mean=0.):
		self.distortion_list = distortion_list

	def __call__(self, img):
		image = np.array(img)
		self.std = self.distortion_list[np.random.choice(len(self.distortion_list), 1)[0]]
		blur = cv2.GaussianBlur(image, (4*self.std+1, 4*self.std+1), self.std, None, self.std, cv2.BORDER_CONSTANT)
		return Image.fromarray(blur) 

	def __repr__(self):
		return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class AddBlurNoise(object):
  def __init__(self, blur_list, noise_list, mean=0.):
    self.blur_list = blur_list
    self.noise_list = noise_list

  def __call__(self, img):
    
    r = np.random.choice(2, 1)[0]

    if(r == 0):
      image = np.array(img)
      self.std = self.blur_list[np.random.choice(len(self.blur_list), 1)[0]]
      blur = cv2.GaussianBlur(image, (4*self.std+1, 4*self.std+1), self.std, None, self.std, cv2.BORDER_CONSTANT)
      return Image.fromarray(blur) 

    else:
      image = np.array(img)
      self.std = self.noise_list[np.random.choice(len(self.noise_list), 1)[0]]
      noise_img = image + np.random.normal(0, self.std, (image.shape[0], image.shape[1], image.shape[2]))
      return Image.fromarray(np.uint8(noise_img)) 


  def __repr__(self):
    return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def save_idx(train_idx, val_idx, savePath):
  data = np.array([train_idx, val_idx])
  np.save(savePath, data)

class MapDataset(torch.utils.data.Dataset):
  def __init__(self, dataset, transformation):
    self.dataset = dataset
    self.transformation = transformation

  def __getitem__(self, index):
    x = self.transformation(self.dataset[index][0])
    y = self.dataset[index][1]
    return x, y

  def __len__(self):
    return len(self.dataset)


def load_caltech(root_path, transf_train, transf_valid, batch_size,savePath_idx_dataset, split_train=0.8):

  dataset = datasets.ImageFolder(root_path)

  train_dataset = MapDataset(dataset, transf_train)
  val_dataset = MapDataset(dataset, transf_valid)

  if (os.path.exists(savePath_idx_dataset)):
    data = np.load(savePath_idx_dataset, allow_pickle=True)
    train_idx, valid_idx = data[0], data[1]

  else:
    nr_samples = len(dataset)
    indices = list(range(nr_samples))
    split = int(np.floor(split_train * nr_samples))
    np.random.shuffle(indices)
    train_idx, valid_idx = indices[:split], indices[split:]
    save_idx(train_idx, valid_idx, savePath_idx_dataset)


  train_data = torch.utils.data.Subset(train_dataset, indices=train_idx)
  val_data = torch.utils.data.Subset(val_dataset, indices=valid_idx)

  trainLoader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, 
                                            num_workers=4)
  valLoader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, 
                                            num_workers=4)

  return trainLoader, valLoader


def trainBranches(model, train_loader, optimizer, criterion, n_branches, epoch, device, loss_weights):
  running_loss = []
  train_acc_dict = {i: [] for i in range(1, (n_branches+1)+1)}
  model.train()

  for i, (data, target) in enumerate(tqdm(train_loader), 1):
    #print("Batch: %s/%s"%(i, len(train_loader)))
    data, target = data.to(device), target.long().to(device)

    output_list, conf_list, class_list = model(data)

    optimizer.zero_grad()
    loss = 0
    for j, (output, inf_class, weight) in enumerate(zip(output_list, class_list, loss_weights), 1):
      loss += weight*criterion(output, target)
      train_acc_dict[j].append(100*inf_class.eq(target.view_as(inf_class)).sum().item()/target.size(0))


    running_loss.append(float(loss.item()))
    loss.backward()
    optimizer.step()
    

    # clear variables
    del data, target, output_list, conf_list, class_list
    torch.cuda.empty_cache()

  loss = round(np.average(running_loss), 4)
  print("Epoch: %s"%(epoch))
  print("Train Loss: %s"%(loss))

  result_dict = {"epoch":epoch, "train_loss": loss}
  for key, value in train_acc_dict.items():
    result_dict.update({"train_acc_branch_%s"%(key): round(np.average(train_acc_dict[key]), 4)})    
    print("Train Acc Branch %s: %s"%(key, result_dict["train_acc_branch_%s"%(key)]))
  
  return result_dict

def evalBranches(model, val_loader, criterion, n_branches, epoch, device):
  running_loss = []
  val_acc_dict = {i: [] for i in range(1, (n_branches+1)+1)}
  model.eval()

  for i, (data, target) in enumerate(val_loader, 1):
    data, target = data.to(device), target.long().to(device)

    output_list, conf_list, class_list = model(data)

    loss = 0
    for j, (output, inf_class, weight) in enumerate(zip(output_list, class_list, loss_weights), 1):
      loss += weight*criterion(output, target)
      val_acc_dict[j].append(100*inf_class.eq(target.view_as(inf_class)).sum().item()/target.size(0))


    running_loss.append(float(loss.item()))    

    # clear variables
    del data, target, output_list, conf_list, class_list
    torch.cuda.empty_cache()

  loss = round(np.average(running_loss), 4)
  print("Epoch: %s"%(epoch))
  print("Val Loss: %s"%(loss))

  result_dict = {"epoch":epoch, "val_loss": loss}
  for key, value in val_acc_dict.items():
    result_dict.update({"val_acc_branch_%s"%(key): round(np.average(val_acc_dict[key]), 4)})    
    print("Val Acc Branch %s: %s"%(key, result_dict["val_acc_branch_%s"%(key)]))
  
  return result_dict


parser = argparse.ArgumentParser(description='Evaluating DNNs perfomance using distorted image: blur ou gaussian noise')
parser.add_argument('--distortion_type', type=str, default="pristine", 
  choices=['pristine', 'gaussian_blur','gaussian_noise', "blur_noise"], help='Distortion Type (default: pristine)')
parser.add_argument('--root_path', type=str, help='Path to the pristine Caltech256-dataset')
#parser.add_argument('--dataset_path', type=str, help='Path to the Caltech-256 dataset')

args = parser.parse_args()



root_dir = args.root_path
seed = 42
distortion_type = "gaussian_blur"
dataset_path = os.path.join(".", "dataset", "256_ObjectCategories")
batch_size = 32
model_name = "mobilenet"
pretrained = True
dataset_name = "caltech"
model_id = 21

mean, std = [0.457342265910642, 0.4387686270106377, 0.4073427106250871],[0.26753769276329037, 0.2638145880487105, 0.2776826934044154]


#if (distortion_type == "gaussian_blur"):
#  distortion_list = [1, 2, 3, 4, 5]
#  distortion_app = AddGaussianBlur
#else:
#  distortion_list = [5, 10, 20, 30, 40]
#  distortion_app = AddGaussianNoise

distortion_app = AddBlurNoise
blur_list = [1, 2, 3, 4, 5]
noise_list = [5, 10, 20, 30, 40]

root_dir = os.path.join(root_dir, model_name, dataset_name)

model_save_path = os.path.join(".", "all_distortion_distorted_model_%s_%s_%s.pth"%(model_name, dataset_name, model_id))
#savePath_idx_dataset = os.path.join(root_dir, "save_idx_b_%s_%s_%s.npy"%(model_name, dataset_name, model_id))
savePath_idx_dataset = os.path.join(".", "save_idx_b_%s_%s_%s.npy"%(model_name, dataset_name, model_id))
#pristine_model_path = os.path.join(root_dir, "pristine_model_b_mobilenet_caltech_21.pth")
pristine_model_path = os.path.join(".", "pristine_ee_model_mobilenet_3_branches_id_1.pth")
df_history_save_path = os.path.join(root_dir, "history_distorted_%s_%s_%s_%s.csv"%(distortion_type, model_name, dataset_name, model_id))


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
np.random.seed(seed)
torch.manual_seed(seed)

distorted_transf_train = transforms.Compose([transforms.Resize((300, 300)),
                                             transforms.RandomApply([distortion_app(blur_list, noise_list, 0)], p=0.5),
                                             transforms.RandomChoice([
                                                                      transforms.ColorJitter(brightness=(0.80, 1.20)),
                                                                      transforms.RandomGrayscale(p = 0.25)]),
                                             transforms.RandomHorizontalFlip(p = 0.25),
                                             transforms.RandomRotation(25),
                                             transforms.ToTensor(), 
                                             transforms.Normalize(mean = mean, std = std),])
        
distorted_transf_valid = transforms.Compose([
                                             transforms.Resize(330), 
                                             transforms.CenterCrop(300),
                                             transforms.RandomApply([distortion_app(blur_list, noise_list, 0)], p=0.5), 
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean = mean, std = std),])


train_loader, val_loader = load_caltech(dataset_path, distorted_transf_train, distorted_transf_valid, 
		batch_size, savePath_idx_dataset)

n_classes = 258
pretrained = True
calibration = True
n_branches = 3
img_dim = 300
exit_type = None
optimizer_name = "Adam"
lr = [1.5e-4, 1e-2]
weight_decay = 0.0005

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

branchynet = B_MobileNet(n_classes, False, n_branches, img_dim, exit_type, device)
branchynet.load_state_dict(torch.load(pristine_model_path)["model_state_dict"])
branchynet = branchynet.to(device)

criterion = nn.CrossEntropyLoss()

for param in branchynet.stages.parameters():
  param.requires_grad = False

optimizer = optim.Adam([{'params': branchynet.exits.parameters(), 'lr': lr[1]},
                        {'params': branchynet.fully_connected.parameters(), 'lr': lr[1]}], weight_decay=weight_decay)


scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 10, eta_min=0, last_epoch=-1)

loss_weights = [1, 1, 1, 1]

epoch = 0
count = 0
best_val_loss = np.inf
patience = 10

df = pd.DataFrame()
while 1:
  epoch+=1
  print("Epoch: %s"%(epoch))
  result = {}
  result.update(trainBranches(branchynet, train_loader, optimizer, criterion, n_branches, epoch, device, loss_weights))
  scheduler.step()
  result.update(evalBranches(branchynet, val_loader, criterion, n_branches, epoch, device))

  df = df.append(pd.Series(result), ignore_index=True)
  #df.to_csv(history_save_path)

  if (result["val_loss"] < best_val_loss):
    best_val_loss = result["val_loss"]
    count = 0
    save_dict = {"model_state_dict": branchynet.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                 "epoch": epoch, "val_loss": result["val_loss"]}
    
    for i in range(1, n_branches+1+1):
      save_dict.update({"val_acc_branch_%s"%(i): result["val_acc_branch_%s"%(i)]})

    torch.save(save_dict, model_save_path)

  else:
    count += 1
    print("Count: %s"%(count))
    if (count > patience):
      print("Stop! Patience is finished")
      break
