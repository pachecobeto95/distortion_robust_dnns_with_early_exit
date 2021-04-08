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

def sendData(filePath, p_tar, nr_samples_edge_branch2, nr_samples_edge_branch3, distortion_type, distortion_lvl):
	url = config.URL_EDGE + "/api/edge/edge_emulator"
	
	data_dict = {"p_tar": p_tar, "nr_branch_2": nr_samples_edge_branch2, "nr_branch_3": nr_samples_edge_branch3,
	"distortion_type": distortion_type, "distortion_lvl": distortion_lvl}

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


def computeNrEdge(distortedData, total_size, p_tar, distortion_lvl):
	distortedData = distortedData[(distortedData.p_tar==p_tar) & (distortedData.distortion_lvl==distortion_lvl)]
	edge_rate_branch_2 = distortedData.edge_exit_rate_branch_2.values[0]/100
	edge_rate_branch_3 = distortedData.edge_exit_rate_branch_3.values[0]/100	

	nr_samples_edge_branch2 = round(edge_rate_branch_2*total_size) 
	nr_samples_edge_branch3 = round(edge_rate_branch_3*total_size)

	return nr_samples_edge_branch2, nr_samples_edge_branch3 
	

def sendDistortedImage(test_loader, distortedData, distortion, distortion_type, p_tar, distortion_lvl, datasetPath):
	dataset_dir_list = os.listdir(datasetPath)

	nr_samples_edge_branch2, nr_samples_edge_branch3 = computeNrEdge(distortedData, len(dataset_dir_list), p_tar, distortion_lvl)
	
	for i, img in enumerate(dataset_dir_list, 1):
		filePath = os.path.join(datasetPath, img)
		sendData(filePath, p_tar, nr_samples_edge_branch2, nr_samples_edge_branch3, distortion_type, distortion_lvl)
		#sys.exit()



def inferenceTimeExp(test_loader, distortedData, distortion, distortion_type, p_tar_list, distortion_list, datasetPath):
	for p_tar in p_tar_list:
		print("P_tar: %s"%(p_tar))

		for distortion_lvl in distortion_list:
			print("Distortion Level: %s"%(distortion_lvl))
			datasetPath = os.path.join(datasetPath, str(distortion_lvl))
			sendDistortedImage(test_loader, distortedData, distortion, distortion_type, p_tar, distortion_lvl, datasetPath)

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

df_pristine = pd.read_csv(os.path.join(".", "results", "result_model_calib_pristine_b_mobilenet_eval_pristine_21_2.csv"))
df_blur = pd.read_csv(os.path.join(".", "results", "result2_model_calib_gaussian_blur_b_mobilenet_eval_gaussian_blur_21_freezing_3.csv"))
df_noise = pd.read_csv(os.path.join(".", "results", "result_model_calib_gaussian_noise_b_mobilenet_eval_gaussian_noise_21_freezing_3.csv"))


if (args.distortion_type == "gaussian_blur"):
	distortion_list = blur_list
	df_data = df_blur

elif (args.distortion_type == "gaussian_noise"):
	distortion_list = noise_list
	df_data = df_noise

else:
	distortion_list = pristine_list
	df_data = df_pristine

dataset = LoadDataset(input_dim, batch_size_test, normalization=normalization)
dataset.set_idx_dataset(config.save_idx_path)
test_loader = dataset.caltech(config.dataset_path)

imgConverter = ImageDistortion(args.distortion_type)
distortion = imgConverter.applyDistortion()

datasetPath = os.path.join(".", "dataset", "distorted_dataset", "Caltech", args.distortion_type)

inferenceTimeExp(test_loader, df_data, distortion, args.distortion_type, p_tar_list, distortion_list, datasetPath)



