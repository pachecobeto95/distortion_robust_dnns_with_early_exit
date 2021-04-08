import os, config, time
import requests, sys, json, os
import numpy as np
from PIL import Image
import pandas as pd
import argparse
from utils import LoadDataset, ImageDistortion


"""
End Device Routine to test Edge device and cloud server
this file sends a distorted image
"""

def sendData(filePath, p_tar, distortion_type, distortion_lvl):
	url = config.URL_EDGE + "/api/edge/recognition_cache"
	data_dict = {"p_tar": 0.8, "distortion_type": distortion_type, "distortion_lvl": distortion_lvl}

	files = [
	('img', (filePath, open(filePath, 'rb'), 'application/octet')),
	('data', ('data', json.dumps(data_dict), 'application/json')),]
	

	status = True	
	while (status):
		r = requests.post(url, files=files, timeout=5)
		if (r.status_code != 200):
			time.sleep(2)
		else:
			status = False


def sendDistortedImage(test_loader, distortion_type, distortion, p_tar, distortion_lvl, datasetPath):
	dataset_dir_list = os.listdir(datasetPath)
	url = config.URL_EDGE + "/api/edge/recognition_cache"

	for i, img in enumerate(dataset_dir_list, 1):
		filePath = os.path.join(datasetPath, img)
		sendData(filePath, p_tar, distortion_type, distortion_lvl)
		#sys.exit()
		if(i >= 100):
			sys.exit()



def inferenceTimeExp(test_loader, distortion_type, distortion, p_tar_list, distortion_list, datasetPath):
	for p_tar in p_tar_list:
		print("P_tar: %s"%(p_tar))

		for distortion_lvl in distortion_list:
			print("Distortion Level: %s"%(distortion_lvl))
			distorted_datasetPath = os.path.join(datasetPath, str(distortion_lvl))
			sendDistortedImage(test_loader, distortion_type, distortion, p_tar, distortion_lvl, distorted_datasetPath)

# Input Arguments. Hyperparameters configuration
parser = argparse.ArgumentParser(description='Evaluating DNNs perfomance using distorted image: blur ou gaussian noise')
parser.add_argument('--distortion_type', type=str, default="pristine", 
  choices=['pristine', 'gaussian_blur','gaussian_noise'], help='Distortion Type (default: pristine)')

args = parser.parse_args()


input_dim = 300
batch_size_test = 1
normalization = False
p_tar_list = [0.7, 0.75, 0.8, 0.85, 0.9]
blur_list = [0, 1, 2, 3, 4, 5]
noise_list = [0, 5, 10, 20, 30, 40]
pristine_list = [0]

if (args.distortion_type == "gaussian_blur"):
	distortion_list = blur_list
elif (args.distortion_type == "gaussian_noise"):
	distortion_list = noise_list
else:
	distortion_list = pristine_list


dataset = LoadDataset(input_dim, batch_size_test, normalization=normalization)
dataset.set_idx_dataset(config.save_idx_path)
test_loader = dataset.caltech(config.dataset_path)

imgConverter = ImageDistortion(args.distortion_type)
distortion = imgConverter.applyDistortion()

datasetPath = os.path.join(".", "dataset", "distorted_dataset", "Caltech", args.distortion_type)

inferenceTimeExp(test_loader, args.distortion_type, distortion, p_tar_list, distortion_list, datasetPath)



