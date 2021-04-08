import os, config
import requests, sys, json, os
import numpy as np
from PIL import Image
import pandas as pd
import argparse
from utils import LoadDataset, ImageDistortion

# Input Arguments. Hyperparameters configuration

def distortionConvertion(test_loader, distortion, distortion_type, savePath):

	for distortion_lvl in distortion_list:
		print("Distortion Level: %s"%(distortion_lvl))
		for i, (data, _) in enumerate(test_loader, 1):
			distorted_img = distortion(data, distortion_lvl)
			img_pil = transforms.ToPILImage()(distorted_img[0])
			img_pil.save(os.path.join(savePath, "%s.jpg"%(i)))
			sys.exit()



parser = argparse.ArgumentParser(description='Evaluating DNNs perfomance using distorted image: blur ou gaussian noise')
parser.add_argument('--distortion_type', type=str, default="pristine", 
  choices=['pristine', 'gaussian_blur','gaussian_noise'], help='Distortion Type (default: pristine)')

args = parser.parse_args()

input_dim = 300
batch_size_test = 1
normalization = False
blur_list = [0, 1, 2, 3, 4, 5]
noise_list = [0, 5, 10, 20, 30, 40]
pristine_list = [0]

dataset = LoadDataset(input_dim, batch_size_test, normalization=normalization)
dataset.set_idx_dataset(config.save_idx_path)
test_loader = dataset.caltech(config.dataset_path)

imgConverter = ImageDistortion(args.distortion_type, normalization=False)
distortion = imgConverter.applyDistortion()

if (args.distortion_type == "gaussian_blur"):
	distortion_list = blur_list
elif (args.distortion_type == "gaussian_noise"):
	distortion_list = noise_list
else:
	distortion_list = pristine_list


savePath = os.path.join(".", "dataset", "distorted_dataset", args.distortion_type)
distortionConvertion(test_loader, distortion, distortion_list, savePath)