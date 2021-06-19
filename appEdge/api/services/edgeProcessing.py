from flask import jsonify, session, current_app as app
import os, pickle, requests, sys, config, time
import numpy as np, json
import torchvision.models as models
import torch
#import cv2
import datetime
import time, io
import pandas as pd
from .utils import loadDistortionClassifier, inferenceTransformationDistortionClassifier
from .utils import init_b_mobilenet, select_distorted_model, inferenceTransformation, read_temperature, BranchesModelWithTemperature
from .utils import compute_p_tar_list, NetworkConfiguration
import torchvision.transforms as transforms
from torchvision import datasets, transforms
from PIL import Image
from .mobilenet import B_MobileNet
from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler


distortion_classifier = loadDistortionClassifier().cuda()
distorted_model_list, distortion_classes = init_b_mobilenet()
distorted_temp_list = read_temperature()
net_config = NetworkConfiguration()

def transform_image(image_bytes):
	my_transforms = transforms.Compose([transforms.Resize(330),
		transforms.CenterCrop(300),
		transforms.ToTensor(),
		transforms.Normalize([0.457342265910642, 0.4387686270106377, 0.4073427106250871], [0.26753769276329037, 0.2638145880487105, 0.2776826934044154])])

	image = Image.open(io.BytesIO(image_bytes))
	return my_transforms(image).unsqueeze(0)


def dnnInference(fileImg, end_dist_type, distortion_lvl, robust):
	url = "%s/api/cloud/cloudProcessing"%(config.URL_CLOUD)
	image_bytes = fileImg.read()
	device = torch.device("cpu")
	tensor = transform_image(image_bytes).to(device)

	img_np = np.array(Image.open(io.BytesIO(image_bytes)))
	response_request = {"status": "ok"}

	dist_prop = "robust" if robust else "standard"
	start = time.time()
	if (robust):
		distortion_type = distortionClassifierInference(img_np, distortion_classifier).item()
	else:
		distortion_type = -1

	#distortion_type = distortionClassifierInference(img_np, distortion_classifier)
	output, conf_list, infer_class = b_mobileNetInferenceEdge(tensor, distorted_model_list, distorted_temp_list, distortion_type, device)

	if (infer_class is None):
		response_request = sendToCloud(url, output, conf_list, distortion_type)
	inference_time = time.time() - start
	if (response_request["status"]=="ok"):
		saveInferenceTime(inference_time, distortion_classes[distortion_type.item()], end_dist_type, distortion_lvl, dist_prop)

	return response_request


def distortionClassifierInference(img, distortionClassifierModel, device):
	softmax = torch.nn.Softmax(dim=1)
	img_fft = inferenceTransformationDistortionClassifier(img)

	distortion_output = distortion_classifier(img_fft).to(device)
	_, distortion_type = torch.max(softmax(distortion_output), 1)

	return distortion_type

def b_mobileNetInferenceEdge(tensor, distorted_model_list, distorted_temp_list, distortion_type):
	#distortedModel, temperature = select_distorted_model(pristine_model, blur_model, noise_model, distortion_type, pristine_temp, blur_temp, noise_temp)
	distortedModel = distorted_model_list[distortion_type].to(device)
	temperature = distorted_temp_list[distortion_type]
	scaled_model = BranchesModelWithTemperature(distortedModel, temperature)
	scaled_model.eval()
	with torch.no_grad():
		output, conf_list, infer_class = scaled_model(tensor.float(), p_tar=config.P_TAR)
	
	return output, conf_list, infer_class


def sendToCloud(url, feature_map, conf_list, distortion_type):
	"""
	This functions sends output data from a partitioning layer from edge device to cloud server.
	This function also sends the info of partitioning layer to the cloud.
	Argments:

	feature_map (Tensor): output data from partitioning layer
	partitioning_layer (int): partitioning layer decided by the optimization method. 
	"""

	data = {'feature': feature_map.detach().cpu().numpy().tolist(), "distortion_type": distortion_type, "conf": conf_list}

	try:
		r = requests.post(url, json=data, timeout=30)
	except requests.exceptions.ConnectTimeout:
		return {"status": "error"}

	if (r.status_code != 200 and r.status_code != 201):
		return {"status": "error"}

	else:
		return {"status": "ok"}

def saveInferenceTime(inference_time, distortion_class, end_dist_type, distortion_lvl):

	#result = {"inference_time": inference_time, "distortion_type": distortion_class,
	#"distortion_lvl": distortion_lvl, "bandwidth": net_config.bandwidth, "latency":net_config.latency, "city":net_config.city}
	result = {"inference_time": inference_time, "distortion_type": distortion_class, "distortion_lvl": distortion_lvl}
	result_path = os.path.join(config.RESULTS_INFERENCE_TIME_EDGE, "%s_inference_time_%s.csv"%(dist_prop, end_dist_type))

	if (not os.path.exists(result_path)):
		df = pd.DataFrame()
	else:
		df = pd.read_csv(result_path)
		df = df.loc[:, ~df.columns.str.contains('^Unnamed')] 

	
	df = df.append(pd.Series(result), ignore_index=True)
	df.to_csv(result_path)


def get_nr_samples_edge(end_dist_type, distortion_lvl, dist_prop):
	nr_samples_edge = 0
	result_path = os.path.join(config.RESULTS_INFERENCE_TIME_EDGE, "%s_inference_time_%s.csv"%(dist_prop, end_dist_type))


	if (os.path.exists(result_path)):
		df_inference_time = pd.read_csv(result_path)
		#df_current = df_inference_time[(df_inference_time.latency==net_config.latency) & (df_inference_time.bandwidth==net_config.bandwidth) & (df_inference_time.distortion_lvl == distortion_lvl)]
		df_current = df_inference_time[df_inference_time.distortion_lvl == distortion_lvl]
		nr_samples_edge = len(df_current.index)

	return nr_samples_edge

def b_mobileNetInferenceEdgeEmulation(tensor, nr_samples_edge, nr_branch_2, nr_branch_3, distorted_model_list, distorted_temp_list, distortion_type, device):
	distortedModel = distorted_model_list[distortion_type].to(device)
	temperature = distorted_temp_list[distortion_type]
	p_tar_list = compute_p_tar_list(nr_samples_edge, nr_branch_2, nr_branch_3)

	distortedModel.eval()
	with torch.no_grad():
		output, conf_list, infer_class = distortedModel.forwardEmulation(tensor.float(), p_tar_list=p_tar_list)

	return output, conf_list, infer_class



def dnnInferenceEmulation(fileImg, nr_branch_2, nr_branch_3, end_dist_type, distortion_lvl, robust):
	url = "%s/api/cloud/cloudProcessingEmulation"%(config.URL_CLOUD)
	image_bytes = fileImg.read()
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	tensor = transform_image(image_bytes).to(device)
	img_np = np.array(Image.open(io.BytesIO(image_bytes)))
	response_request = {"status": "ok"}
	dist_prop = "robust" if robust else "standard"


	nr_samples_edge = get_nr_samples_edge(end_dist_type, distortion_lvl, dist_prop)

	start = time.time()
	if (robust):
		distortion_type = distortionClassifierInference(img_np, distortion_classifier, device).item()
	else:
		distortion_type = -1

	output, conf_list, infer_class = b_mobileNetInferenceEdgeEmulation(tensor, nr_samples_edge, nr_branch_2, nr_branch_3, distorted_model_list, distorted_temp_list, distortion_type, device)

	if (infer_class is None):
		response_request = sendToCloud(url, output, conf_list, distortion_classes.index(end_dist_type))

	inference_time = time.time() - start
	if (response_request["status"] == "ok"):
		saveInferenceTimeEmulation(inference_time, distortion_classes[distortion_type], end_dist_type, distortion_lvl, dist_prop)

	return response_request

def saveInferenceTimeEmulation(inference_time, distortion_class, end_dist_type, distortion_lvl, dist_prop):
	
	result = {"inference_time": inference_time, "distortion_type": distortion_class,
	"distortion_lvl": distortion_lvl}

	result_path = os.path.join(config.RESULTS_INFERENCE_TIME_EDGE, "%s_inference_time_%s.csv"%(dist_prop, end_dist_type))

	if (not os.path.exists(result_path)):
		df = pd.DataFrame()
	else:
		df = pd.read_csv(result_path)	
		df = df.loc[:, ~df.columns.str.contains('^Unnamed')] 
	
	df = df.append(pd.Series(result), ignore_index=True)
	df.to_csv(result_path) 
