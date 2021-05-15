import os
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
from torchvision import datasets
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.utils import save_image
import os, cv2
from torchvision import transforms, utils, datasets
from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler
import copy
import matplotlib.pyplot as plt
from load_datasets import LoadDataset
from PIL import Image

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


class QualityConverter():
  def __init__(self, dataLoader, dataset_name, savePath):
    """
    Converts a dataset into distorted dataset with different distortion types and levels.

    Arguments are

    * dataLoader:                         contains the dataset and the classes of each image
    * dataset_name:                       the dataset name
    * savePath                            path to save the distorted images
    """

    self.dataLoader = dataLoader
    self.dataset_name = dataset_name
    self.savePath = savePath

  def __applyGaussianBlur(self, img, blur_std):
    """
    this method adds gaussian blur into the images
    img (tensor):      image in tensor format, whose shape is (batch_size, channel, width, height)
    blur_std (int):    stadard deviation used to add blur into the image
    return:
    img_pil (PIL Image):    image in PIL object
    """
    img_pil = transforms.ToPILImage()(img[0]) # img[0] removes batch_size-related component and transforms in PIL Image
    blur = cv2.GaussianBlur(np.array(img_pil), (4*blur_std+1, 4*blur_std+1),blur_std, None, blur_std, cv2.BORDER_CONSTANT)
    img_pil = Image.fromarray(np.uint8(blur),  'RGB')
    return img_pil    
    #return cv2.GaussianBlur(img,(4*blur_std+1, 4*blur_std+1),blur_std, None, blur_std, cv2.BORDER_CONSTANT)

  def __applyGaussianNoise(self, img_batch, noise_lvl):
    # this method adds Additive White Gaussian Noise into the images
    noise_img_batch = img_batch
    if (noise_lvl > 0):
      noise_img_batch = []
      for img in img_batch:
        img_np = np.array(transforms.ToPILImage()(img))
        noise_img = img_np + np.random.normal(0, noise_lvl, (img_np.shape[0], img_np.shape[1], img_np.shape[2]))

        noise_img_batch.append(transforms.ToTensor()(np.uint8(noise_img)))
      noise_img_batch = torch.stack(noise_img_batch)
    return transforms.ToPILImage()(noise_img_batch.float()[0])

  def __randomAngle(self):
    possible_angles = [0, 30, 45, 60, 90, 120, 135, 150, 180,
                       210, 240, 270, 300, 315, 330]
    angle_idx = np.random.choice(len(possible_angles), 1)[0]
    return possible_angles[angle_idx]

  def __applyMotionBlur(self, img, kernel_size):
    img_pil = transforms.ToPILImage()(img[0])
    imgarray = np.array(img_pil, dtype="float32")
    angle = self.__randomAngle()
    m = cv2.getRotationMatrix2D((kernel_size/2, kernel_size/2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(kernel_size))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, m, (kernel_size, kernel_size))
    motion_blur_kernel = motion_blur_kernel/kernel_size
    blurred = cv2.filter2D(imgarray, -1, motion_blur_kernel)
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    return Image.fromarray(np.uint8(blurred), "RGB")

  def gausianBlur(self, distortion_lvl):
    self.distortion_type = "blur"
    self.distortion_lvl = distortion_lvl
    distortion = self.__applyGaussianBlur
    self.__generate_distorted_dataset(distortion)

  def motionBlur(self, distortion_lvl):
    self.distortion_type = "motion_blur"
    self.distortion_lvl = distortion_lvl
    distortion = self.__applyMotionBlur
    self.__generate_distorted_dataset(distortion)

  def whiteGaussianNoise(self, distortion_lvl):
    self.distortion_type = "noise"
    self.distortion_lvl = distortion_lvl
    distortion = self.__applyGaussianNoise
    self.__generate_distorted_dataset(distortion)

  def pristine(self):
    self.distortion_type = "pristine"
    self.distortion_lvl = [0]
    distortion = self.__applyPristine
    self.__generate_distorted_dataset(distortion)
    

  def apply_fourier_transformation(self, img):
    gray_img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
    dft = np.fft.fft2(gray_img)
    fshift = np.fft.fftshift(dft)
    magnitude_spectrum = 20*np.log(np.abs(fshift))
    result = Image.fromarray((magnitude_spectrum).astype(np.uint8))
    return result
    
  def __generate_distorted_dataset(self, distortion, end=10000, log_interval=100):
    """
    This method 
    Arguments are

    * distortion:         a function that adds distortion in the images 
    """
    # if the distortion_level parameter is a int, this converts this integer to a list.
    if (isinstance(self.distortion_lvl, int)):
      self.distortion_lvl = list([self.distortion_lvl])

    # this converts the dataset for different levels of a distortion type. 
    for i, (img, label) in enumerate(self.dataLoader):
      print(i)
      dist_lvl = self.distortion_lvl[np.random.choice(len(self.distortion_lvl), 1)[0]]
      finalSavePath = os.path.join(self.savePath, self.distortion_type)
      
      if (not os.path.exists(finalSavePath)):
        os.makedirs(finalSavePath)

      distorted_img = distortion(img, dist_lvl)
      dft_img = self.apply_fourier_transformation(distorted_img)
      dft_img.save(os.path.join(finalSavePath, "%s.jpg"%(i)))
      if (i >= end):
        break


parser = argparse.ArgumentParser(description='Evaluating DNNs perfomance using distorted image: blur ou gaussian noise')
parser.add_argument('--distortion_type', type=str, default="pristine", 
  choices=['pristine', 'gaussian_blur','gaussian_noise'], help='Distortion Type (default: pristine)')
parser.add_argument('--root_path', type=str, help='Path to the pristine Caltech256-dataset')
parser.add_argument('--save_path', type=str, help='Path to save the Fourier spectrum')

args = parser.parse_args()


input_dim = 224
batch_size_train, batch_size_test = 1, 1
dataset = LoadDataset(input_dim, batch_size_train, batch_size_test)
trainLoader, _, _ = dataset.customDataset(args.root_path, split_train=0.8)
quality_converter = QualityConverter(trainLoader, "Caltech256", args.save_path)
blur_lvl_list = [1, 2, 3, 4, 5]
noise_lvl_list = [5, 10, 20, 30, 40]

if (args.distortion_type == "gaussian_blur"):
  quality_converter.gausianBlur(blur_lvl_list)

elif (args.distortion_type == "gaussian_noise"):
  quality_converter.whiteGaussianNoise(noise_lvl_list)

else:
  quality_converter.pristine()

