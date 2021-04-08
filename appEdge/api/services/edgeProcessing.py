from flask import jsonify, session, current_app as app
import os, pickle, requests, sys, config, time
import numpy as np, json
import torchvision.models as models
import torch
import cv2
import datetime
import time, io
import pandas as pd
from .utils import loadDistortionClassifier, inferenceTransformationDistortionClassifier
from .utils import init_b_mobilenet, select_distorted_model, inferenceTransformation, read_temperature, BranchesModelWithTemperature
from .utils import compute_p_tar_list
import torchvision.transforms as transforms
from torchvision import datasets, transforms
from PIL import Image
from .mobilenet import B_MobileNet
from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler


distortion_classifier = loadDistortionClassifier()
distorted_model_list, distortion_classes = init_b_mobilenet()
distorted_temp_list = read_temperature()

def transform_image(image_bytes):
	my_transforms = transforms.Compose([transforms.Resize(330),
		transforms.CenterCrop(300),
		transforms.ToTensor(),
		transforms.Normalize([0.457342265910642, 0.4387686270106377, 0.4073427106250871], [0.26753769276329037, 0.2638145880487105, 0.2776826934044154])])

	image = Image.open(io.BytesIO(image_bytes))
	return my_transforms(image).unsqueeze(0)


def dnnInference(fileImg, p_tar, end_dist_type, distortion_lvl):
	url = "%s/api/cloud/cloudProcessing"%(config.URL_CLOUD)
	image_bytes = fileImg.read()
	tensor = transform_image(image_bytes)
	img_np = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)

	start = time.time()
	distortion_type = distortionClassifierInference(img_np, distortion_classifier)
	output, conf_list, infer_class = b_mobileNetInferenceEdge(tensor, p_tar, distorted_model_list, distorted_temp_list, distortion_type)

	if (infer_class is None):
		print("cloud")
		sendToCloud(url, output, conf_list, p_tar, start, distortion_type, distortion_lvl)

	else:
		inference_time = time.time() - start
		saveInferenceTime(inference_time, p_tar, distortion_classes[distortion_type.item()], end_dist_type, distortion_lvl)		
	return {"status": "ok"}


def distortionClassifierInference(img, distortionClassifierModel):
	softmax = torch.nn.Softmax(dim=1)
	img_fft = inferenceTransformationDistortionClassifier(img)

	distortion_output = distortion_classifier(img_fft)
	_, distortion_type = torch.max(softmax(distortion_output), 1)

	return distortion_type

def b_mobileNetInferenceEdge(tensor, p_tar, distorted_model_list, distorted_temp_list, distortion_type):
	#distortedModel, temperature = select_distorted_model(pristine_model, blur_model, noise_model, distortion_type, pristine_temp, blur_temp, noise_temp)
	distortedModel = distorted_model_list[distortion_type]
	temperature = distorted_temp_list[distortion_type]
	scaled_model = BranchesModelWithTemperature(distorted_model_list[-1], distorted_temp_list[-1])
	scaled_model.eval()
	with torch.no_grad():
		output, conf_list, infer_class = scaled_model(tensor.float(), p_tar=p_tar)	
	
	return output, conf_list, infer_class


def sendToCloud(url, feature_map, conf_list, p_tar, start, distortion_type, distortion_lvl):
	"""
	This functions sends output data from a partitioning layer from edge device to cloud server.
	This function also sends the info of partitioning layer to the cloud.
	Argments:

	feature_map (Tensor): output data from partitioning layer
	partitioning_layer (int): partitioning layer decided by the optimization method. 
	"""

	data = {'feature': feature_map.detach().numpy().tolist(), "start": start,
	"distortion_type": distortion_type.item(), "conf": conf_list, "p_tar": p_tar, "distortion_lvl":distortion_lvl}

	r = requests.post(url, json=data)
	
	if (r.status_code != 200 and r.status_code != 201):
		return {"status": "error"}

	else:
		return {"status": "ok"}

def saveInferenceTime(inference_time, p_tar, distortion_class, end_dist_type, distortion_lvl):
	
	result = {"inference_time": inference_time,"p_tar": p_tar, "distortion_type": distortion_class,
	"distortion_lvl": distortion_lvl}
	
	result_path = os.path.join(config.RESULTS_INFERENCE_TIME_EDGE, "inference_time_%s.csv"%(end_dist_type))

	if (not os.path.exists(result_path)):
		df = pd.DataFrame()
	else:
		df = pd.read_csv(result_path)	
		df = df.loc[:, ~df.columns.str.contains('^Unnamed')] 
	
	df = df.append(pd.Series(result), ignore_index=True)
	df.to_csv(result_path)


def get_nr_samples_edge(end_dist_type):
	nr_samples_edge = 0
	result_path = os.path.join(config.RESULTS_INFERENCE_TIME_EDGE, "inference_time_emulation_%s.csv"%(end_dist_type))
	if (os.path.exists(result_path)):
		df_inference_time = pd.read_csv(result_path)
		nr_samples_edge = len(df_inference_time.index)

	return nr_samples_edge

def b_mobileNetInferenceEdgeEmulation(tensor, nr_samples_edge, nr_branch_2, nr_branch_3, distorted_model_list, distorted_temp_list, distortion_type):
	distortedModel = distorted_model_list[distortion_type]
	temperature = distorted_temp_list[distortion_type]
	#scaled_model = BranchesModelWithTemperature(distortedModel, temperature)
	print(nr_samples_edge, nr_branch_2, nr_branch_3)
	p_tar_list = compute_p_tar_list(nr_samples_edge, nr_branch_2, nr_branch_3)

	distortedModel.eval()
	with torch.no_grad():
		output, conf_list, infer_class = distortedModel.forwardEmulation(tensor.float(), p_tar_list=p_tar_list)	
	
	return output, conf_list, infer_class



def dnnInferenceEmulation(fileImg, p_tar, nr_branch_2, nr_branch_3, end_dist_type, distortion_lvl):
	url = "%s/api/cloud/cloudProcessingEmulation"%(config.URL_CLOUD)
	image_bytes = fileImg.read()
	tensor = transform_image(image_bytes)
	img_np = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)

	nr_samples_edge = get_nr_samples_edge(end_dist_type)

	start = time.time()
	distortion_type = distortionClassifierInference(img_np, distortion_classifier)
	output, conf_list, infer_class = b_mobileNetInferenceEdgeEmulation(tensor, nr_samples_edge, nr_branch_2, nr_branch_3, distorted_model_list, distorted_temp_list, distortion_type)
	
	if (infer_class is None):
		sendToCloud(url, output, conf_list, p_tar, start, distortion_type, distortion_lvl)	
	else:
		inference_time = time.time() - start
		saveInferenceTimeEmulation(inference_time, p_tar, distortion_classes[distortion_type.item()], end_dist_type, distortion_lvl)		
	return {"status": "ok"}

def saveInferenceTimeEmulation(inference_time, p_tar, distortion_class, end_dist_type, distortion_lvl):
	
	result = {"inference_time": inference_time,"p_tar": p_tar, "distortion_type": distortion_class,
	"distortion_lvl": distortion_lvl}
	
	result_path = os.path.join(config.RESULTS_INFERENCE_TIME_EDGE, "inference_time_emulation_%s.csv"%(end_dist_type))

	if (not os.path.exists(result_path)):
		df = pd.DataFrame()
	else:
		df = pd.read_csv(result_path)	
		df = df.loc[:, ~df.columns.str.contains('^Unnamed')] 
	
	df = df.append(pd.Series(result), ignore_index=True)
	df.to_csv(result_path) 