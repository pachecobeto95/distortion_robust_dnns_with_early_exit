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
  def __init__(self, distortion_lvl, mean=0.):
    self.distortion_lvl = distortion_lvl

  def __call__(self, img):
    image = np.array(img)
    self.std = self.distortion_lvl
    noise_img = image + np.random.normal(0, self.std, (image.shape[0], image.shape[1], image.shape[2]))
    return Image.fromarray(np.uint8(noise_img)) 
    
  def __repr__(self):
    return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class AddGaussianBlur(object):
	def __init__(self, distortion_lvl, mean=0.):
		self.distortion_lvl = distortion_lvl

	def __call__(self, img):
		image = np.array(img)
		self.std = distortion_lvl
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


def run_inference_data(model, val_loader, n_branches, dist_type_model, dist_type_data, distortion_lvl, device):
  running_loss = []
  conf_branches_list, infered_class_branches_list, target_list, correct_list, entropy_list = [], [], [], [], []

  model.eval()

  with torch.no_grad():
    for i, (data, target) in enumerate(tqdm(val_loader), 1):
      data, target = data.to(device), target.long().to(device)

      _, conf_branches, infered_class_branches = model(data)

      conf_branches_list.append([conf.item() for conf in conf_branches])
      infered_class_branches_list.append([inf_class.item() for inf_class in infered_class_branches])    
      correct_list.append([infered_class_branches[i].eq(target.view_as(infered_class_branches[i])).sum().item() for i in range(n_branches+1)])
      target_list.append(target.item())


      del data, target, conf_branches, infered_class_branches
      torch.cuda.empty_cache()

  conf_branches_list = np.array(conf_branches_list)
  infered_class_branches_list = np.array(infered_class_branches_list)
  correct_list = np.array(correct_list)
  target_list = np.array(target_list)

  results = {"distortion_type_model": [dist_type_model]*len(target_list),
  "distortion_type_data": [dist_type_data]*len(target_list), "distortion_lvl": [distortion_lvl]*len(target_list), 
  "target": target_list}

  for i in range(n_branches+1):
    results.update({"conf_branch_%s"%(i+1): conf_branches_list[:, i],
      "infered_class_branches_%s"%(i+1): infered_class_branches_list[:, i],
      "correct_branch_%s"%(i+1): correct_list[:, i]}) 

  return results



parser = argparse.ArgumentParser(description='Evaluating DNNs perfomance using distorted image: blur ou gaussian noise')
parser.add_argument('--distortion_type_data', type=str, default="pristine", 
  choices=['pristine', 'gaussian_blur','gaussian_noise'], help='Distortion Type (default: pristine)')
parser.add_argument('--distortion_type_model', type=str, default="pristine", 
  choices=['pristine', 'gaussian_blur','gaussian_noise', "blur_noise"], help='Distortion Type (default: pristine)')
parser.add_argument('--root_path', type=str, help='Path to the pristine Caltech256-dataset')

args = parser.parse_args()



root_dir = args.root_path
seed = 42
distortion_type_data = args.distortion_type_data
dataset_path = os.path.join(".", "dataset", "256_ObjectCategories")
model_name = "mobilenet"
pretrained = True
dataset_name = "caltech"
model_id = 21

mean, std = [0.457342265910642, 0.4387686270106377, 0.4073427106250871],[0.26753769276329037, 0.2638145880487105, 0.2776826934044154]


if (distortion_type_data == "gaussian_blur"):
  distortion_list = [1, 2, 3, 4, 5]
  distortion_app = AddGaussianBlur

else:
  distortion_list = [5, 10, 20, 30, 40]
  distortion_app = AddGaussianNoise
#else:
#  blur_list, noise_list = [1, 2, 3, 4, 5], [5, 10, 20, 30, 40]
#  distortion_list = [blur_list, noise_list]
#  distortion_app = AddBlurNoise

root_dir = os.path.join(root_dir, model_name, dataset_name)

#model_save_path = os.path.join(".", "%s_distorted_model_%s_%s_%s.pth"%(distortion_type_model, model_name, dataset_name, model_id))
model_save_path = os.path.join(".", "%s_ee_model_mobilenet_3_branches_id_1.pth"%(args.distortion_type_model))
savePath_idx_dataset = os.path.join(".", "save_idx_b_%s_%s_%s.npy"%(model_name, dataset_name, model_id))
inference_data_path = os.path.join(".", "luis_results.csv")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
np.random.seed(seed)
torch.manual_seed(seed)
n_classes = 258
calibration = True
n_branches = 3
img_dim = 300
exit_type = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

branchynet = B_MobileNet(n_classes, False, n_branches, img_dim, exit_type, device)
branchynet.load_state_dict(torch.load(model_save_path)["model_state_dict"])
branchynet = branchynet.to(device)


for distortion_lvl in distortion_list:

  df = pd.DataFrame()

  distorted_transf_train = transforms.Compose([transforms.Resize((300, 300)),
                                               transforms.RandomApply([distortion_app(distortion_lvl, 0)], p=1),
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
                                               transforms.RandomApply([distortion_app(distortion_lvl, 0)], p=1), 
                                               transforms.ToTensor(),
                                               transforms.Normalize(mean = mean, std = std),])


  train_loader, val_loader = load_caltech(dataset_path, distorted_transf_train, distorted_transf_valid, 
      1, savePath_idx_dataset)


  result = run_inference_data(branchynet, val_loader, n_branches, args.distortion_type_model, distortion_type_data, distortion_lvl, device)
  save_result(result, inference_data_path)
