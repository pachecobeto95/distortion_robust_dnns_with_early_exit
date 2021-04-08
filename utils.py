import os, config
import requests, sys, json, os
import numpy as np
from PIL import Image
import pandas as pd
import cv2
import torch
import torchvision.transforms as transforms
from torchvision import datasets


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



class LoadDataset():
  def __init__(self, input_dim, batch_size_test, normalization=True):
    self.input_dim = input_dim
    self.batch_size_test = batch_size_test
    self.savePath_idx_dataset = None

    mean=[0.457342265910642, 0.4387686270106377, 0.4073427106250871]
    std=[0.26753769276329037, 0.2638145880487105, 0.2776826934044154]

    transformation_valid_list = [transforms.Resize(330), 
                                          transforms.CenterCrop(300), 
                                          transforms.ToTensor()]
    
    if (normalization):
      transformation_train_list.append(transforms.Normalize(mean = mean, std = std))
      transformation_valid_list.append(transforms.Normalize(mean = mean, std = std))

        
    self.transformations_valid = transforms.Compose(transformation_valid_list)

  def set_idx_dataset(self, save_idx_path):
    self.savePath_idx_dataset = save_idx_path


  def caltech(self, root_path, split_train=0.8):

    dataset = datasets.ImageFolder(root_path)

    val_dataset = MapDataset(dataset, self.transformations_valid)


    if (self.savePath_idx_dataset is not None):
      data = np.load(self.savePath_idx_dataset, allow_pickle=True)
      train_idx, valid_idx = data[0], data[1]
      indices = list(range(len(valid_idx)))
      split = int(np.floor(0.5 * len(valid_idx)))
      valid_idx, test_idx = valid_idx[:split], valid_idx[split:]

    else:
      nr_samples = len(dataset)
      indices = list(range(nr_samples))
      split = int(np.floor(split_train * nr_samples))
      np.random.shuffle(indices)
      rain_idx, test_idx = indices[:split], indices[split:]


    test_data = torch.utils.data.Subset(val_dataset, indices=test_idx)

    testLoader = torch.utils.data.DataLoader(test_data, batch_size=self.batch_size_test, 
                                              num_workers=4)


    return testLoader


class ImageDistortion(object):
  def __init__(self, distortion_type, normalization=True):
    self.norm_transformation = transforms.Compose([transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    self.distortion_type = distortion_type
    self.normalization = normalization

  def gaussian_blur(self, img_batch, blur_std):
    blurred_img_batch = img_batch
    if (blur_std > 0):
      blurred_img_batch = []
      for img in img_batch:
        img_np = np.array(transforms.ToPILImage()(img))
        blur = cv2.GaussianBlur(img_np, (4*blur_std+1, 4*blur_std+1), blur_std, None, blur_std, cv2.BORDER_CONSTANT)
        blurred_img_batch.append(transforms.ToTensor()(blur))
      blurred_img_batch = torch.stack(blurred_img_batch)
    

    if (self.normalization):
        return self.norm_transformation(blurred_img_batch)
    else:
        return blurred_img_batch

  def gaussian_noise(self, img_batch, noise_lvl):
    noise_img_batch = img_batch
    if (noise_lvl > 0):
      noise_img_batch = []
      for img in img_batch:
        img_np = np.array(transforms.ToPILImage()(img))
        noise_img = img_np + np.random.normal(0, noise_lvl, (img_np.shape[0], img_np.shape[1], img_np.shape[2]))

        noise_img_batch.append(transforms.ToTensor()(np.uint8(noise_img)))
      noise_img_batch = torch.stack(noise_img_batch)

    if (self.normalization):
        return self.norm_transformation(noise_img_batch).float()
    else:
        return noise_img_batch.float()

  def pristine(self, img_batch, std=None):
    if (self.normalization):
        return self.norm_transformation(img_batch)
    else:
    	return img_batch

  def applyDistortion(self):
    def func_not_found():
      print("No distortion %s is found"%(self.distortion_type))
      sys.exit()
    
    func_name = getattr(self, self.distortion_type, func_not_found)
    return func_name












