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

def sendData(filePath, nr_samples_edge_branch2, nr_samples_edge_branch3, distortion_type, distortion_lvl, robust):
	url = "http://172.17.0.1:5000/api/edge/edge_emulator"

	data_dict = {"nr_branch_2": nr_samples_edge_branch2, "nr_branch_3": nr_samples_edge_branch3,
	"distortion_type": distortion_type, "distortion_lvl": distortion_lvl, "robust": robust}

	#data_dict = {"nr_branch_2": 0, "nr_branch_3": 0,
	#"distortion_type": distortion_type, "distortion_lvl": distortion_lvl, "robust": robust}

	files = [
	('img', (filePath, open(filePath, 'rb'), 'application/octet')),
	('data', ('data', json.dumps(data_dict), 'application/json')),]
	#status = True
	#while (status):
	#	r = requests.post(url, files=files, timeout=150)
	#	if (r.status_code != 200):
	#		time.sleep(2)
	#	else:
	#		status = False
	try:
		r = requests.post(url, files=files, timeout=50)
	except requests.exceptions.ConnectTimeout:
		pass
	#except requests.exceptions.ConnectionError:
	#	pass



def computeNrEdge(distortedData, total_size, distortion_lvl):
	distortedData = distortedData[(distortedData.p_tar==config.P_TAR) & (distortedData.distortion_lvl==distortion_lvl)]
	edge_rate_branch_2 = distortedData.edge_exit_rate_branch_2.values[0]/100
	edge_rate_branch_3 = distortedData.edge_exit_rate_branch_3.values[0]/100	

	nr_samples_edge_branch2 = round(edge_rate_branch_2*total_size) 
	nr_samples_edge_branch3 = round(edge_rate_branch_3*total_size)

	return nr_samples_edge_branch2, nr_samples_edge_branch3 
	
def sendNetworkConfEdge(bandwidth, latency, city):
	url = "http://172.17.0.1:5000/api/edge/edge_update_network_config"
	data_dict = {"bandwidth": bandwidth, "latency": latency, "city": city}
	status = True
	while status:
		r = requests.post(url, json=data_dict, timeout=30)
		if (r.status_code != 200):
			time.sleep(2)
		else:
			status = False

def sendImg(url, imgPath):
	status = True
	files = {"img": open(imgPath, "rb")}
	while status:
		r = requests.post(url, files=files, timeout=5)
		if (r.status_code != 200):
			time.sleep(2)
		else:
			status = False

def starterChannel(datasetPath, nr_imgs=50):
	url = "http://172.17.0.1:5000/api/edge/starter_channel"
	datasetPath = os.path.join(datasetPath, "0")
	dataset_dir_list = os.listdir(datasetPath)[:nr_imgs]
	for img in dataset_dir_list:
		filePath = os.path.join(datasetPath, img)
		sendImg(url, filePath)

def sendNetworkConfCloud(bandwidth, latency):
	url = "http://0.0.0.0:3000/api/cloud/cloud_update_network_config"
	data_dict = {"bandwidth": bandwidth, "latency": latency}
	status = True
	while status:
		r = requests.post(url, json=data_dict, timeout=7)
		if (r.status_code != 200):
			time.sleep(2)
		else:
			status = False

def sendDistortedImage(test_loader, distortedData, distortion, distortion_type, distortion_lvl, datasetPath, savePath, robust):
	dataset_dir_list = os.listdir(datasetPath)

	nr_samples_edge_branch2, nr_samples_edge_branch3 = computeNrEdge(distortedData, len(dataset_dir_list), distortion_lvl)
	
	for i, img in enumerate(dataset_dir_list, 1):
		print(i, len(dataset_dir_list))
		filePath = os.path.join(datasetPath, img)
		#start = time.time()
		sendData(filePath, nr_samples_edge_branch2, nr_samples_edge_branch3, distortion_type, distortion_lvl, robust)
		#end = time.time()
		#rtt_end_device = end-start
		#save_result(rtt_end_device, , savePath)



def inferenceTimeExp(test_loader, bandwidth, latency, distortedData, distortion, distortion_type, distortion_list, datasetPath, savePath, robust=True):

	dist_prop = "robust" if robust else "standard"
	savePathFinal = os.path.join(savePath, "%s_rtt_end_device_%s.csv"%(dist_prop, distortion_type))
	for distortion_lvl in distortion_list:
		print("Distortion Level: %s"%(distortion_lvl))
		datasetPath2 = os.path.join(datasetPath, str(distortion_lvl))
		sendDistortedImage(test_loader, distortedData, distortion, distortion_type, distortion_lvl, datasetPath2, savePathFinal, robust)

# Input Arguments. Hyperparameters configuration
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

df_pristine = pd.read_csv(os.path.join(".", "results", "result_model_calib_pristine_b_mobilenet_eval_pristine_21_1.csv"))
#df_blur = pd.read_csv(os.path.join(".", "results", "result2_model_calib_gaussian_blur_b_mobilenet_eval_gaussian_blur_21_freezing_3.csv"))
df_blur = pd.read_csv("./results/total_model_calib_gaussian_blur_b_mobilenet_eval_gaussian_blur_21.csv")
df_noise = pd.read_csv(os.path.join(".", "results", "result_model_calib_gaussian_noise_b_mobilenet_eval_gaussian_noise_21_freezing_3.csv"))
df_standard = pd.read_csv(os.path.join(".", "results", "result_model_calib_pristine_b_mobilenet_eval_%s_21_1.csv"%(args.distortion_type)))

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

datasetPath = os.path.join(".", "dataset", "distorted_dataset", args.distortion_type)
savePath = os.path.join(".", "inference_time_result_end_device")

latency_list = [5, 90, 108]
bandwidth_list = [93.6, 73.3, 60]
city_list = ["sp", "fremont", "paris"]
#latency_list = [108]
#bandwidth_list = [60]
#city_list = ["paris"]

for bandwidth, latency, city  in zip(bandwidth_list, latency_list, city_list):
	sendNetworkConfEdge(bandwidth, latency, city)
	sendNetworkConfCloud(bandwidth, latency)
	starterChannel(datasetPath)
	inferenceTimeExp(test_loader, bandwidth, latency, df_data, distortion, args.distortion_type, distortion_list, datasetPath, savePath, robust=True)
	inferenceTimeExp(test_loader, bandwidth, latency, df_standard, distortion, args.distortion_type, distortion_list, datasetPath, savePath, robust=False)
	time.sleep(30)

