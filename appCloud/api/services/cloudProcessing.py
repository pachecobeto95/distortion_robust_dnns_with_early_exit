from flask import jsonify, session, current_app as app
import os, pickle, requests, sys, config, time
import numpy as np, json
import torchvision.models as models
import torch
import cv2
import datetime
import time, io
import pandas as pd
import torchvision.transforms as transforms
from torchvision import datasets, transforms
from PIL import Image
from .mobilenet import B_MobileNet
from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler
from .utils import init_b_mobilenet, select_distorted_model, read_temperature 
from .utils import BranchesModelWithTemperature


distorted_model_list, distortion_classes  = init_b_mobilenet()
distorted_temp_list = read_temperature()


def dnnInference(feature, conf_list, start, distortion_type, p_tar, distortion_lvl):
	#try:
	feature = torch.Tensor(feature)
	output, conf_list, inf_class = b_mobileNetInferenceCloud(feature, p_tar, conf_list, distorted_model_list, distortion_type)
	inference_time = time.time() - start
	saveInferenceTime(inference_time, p_tar, distortion_classes[distortion_type], distortion_lvl)


	return {"status": "ok"}

	#except Exception as e:
	#	return {"status": "error"}

def b_mobileNetInferenceCloud(tensor, p_tar, conf_list, distorted_model_list, distortion_type):
	distortedModel = distorted_model_list[distortion_type]
	temperature = distorted_temp_list[distortion_type]
	scaled_model = BranchesModelWithTemperature(distortedModel, temperature)
	scaled_model.eval()

	with torch.no_grad():
		output, conf_list, infer_class = scaled_model(tensor.float(), conf_list, p_tar=p_tar)	
	
	return output, conf_list, infer_class

def saveInferenceTime(inference_time, p_tar, distortion_type, distortion_lvl):
	
	result = {"inference_time": inference_time, "p_tar": p_tar, "distortion_type": distortion_type,
	"distortion_lvl": distortion_lvl}
	
	results_path = os.path.join(config.RESULTS_INFERENCE_TIME_CLOUD, "inference_time.csv")
	if (not os.path.exists(results_path)):
		df = pd.DataFrame()
	else:
		df = pd.read_csv(results_path)
		df = df.loc[:, ~df.columns.str.contains('^Unnamed')] 
	
	
	df = df.append(pd.Series(result), ignore_index=True)
	df.to_csv(results_path)



def dnnInferenceEmulation(feature, conf_list, start, distortion_type, p_tar, distortion_lvl):
	#try:
	feature = torch.Tensor(feature)
	output, conf_list, inf_class = b_mobileNetInferenceCloudEmulation(feature, p_tar, conf_list, distorted_model_list, distortion_type)
	inference_time = time.time() - start
	saveInferenceTimeEmulation(inference_time, p_tar, distortion_classes[distortion_type], distortion_lvl)


	return {"status": "ok"}

	#except Exception as e:
	#	return {"status": "error"}

def saveInferenceTimeEmulation(inference_time, p_tar, distortion_class, distortion_lvl):
	
	result = {"inference_time": inference_time,"p_tar": p_tar, "distortion_type": distortion_class,
	"distortion_lvl": distortion_lvl}
	
	result_path = os.path.join(config.RESULTS_INFERENCE_TIME_CLOUD, "inference_time_emulation_%s.csv"%(distortion_class))

	if (not os.path.exists(result_path)):
		df = pd.DataFrame()
	else:
		df = pd.read_csv(result_path)	
		df = df.loc[:, ~df.columns.str.contains('^Unnamed')] 
	
	df = df.append(pd.Series(result), ignore_index=True)
	df.to_csv(result_path) 

def b_mobileNetInferenceCloudEmulation(tensor, p_tar, conf_list, distorted_model_list, distortion_type):
	distortedModel = distorted_model_list[distortion_type]
	temperature = distorted_temp_list[distortion_type]
	scaled_model = BranchesModelWithTemperature(distortedModel, temperature)
	scaled_model.eval()

	with torch.no_grad():
		output, conf_list, infer_class = scaled_model(tensor.float(), conf_list, p_tar=p_tar)	
	
	return output, conf_list, infer_class