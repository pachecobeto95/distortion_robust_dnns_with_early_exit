import os, pickle, requests, sys, config, time
import numpy as np, json
import torchvision.models as models
import torch, cv2
import torch.nn as nn
import torchvision.transforms as transforms
from .mobilenet import B_MobileNet
from PIL import Image
import pandas as pd

def load_model(model, modelPath, device):
	model.load_state_dict(torch.load(modelPath, map_location=device)["model_state_dict"])	
	return model.to(device)


def init_b_mobilenet():
	n_classes = 258
	img_dim = 300
	exit_type = None
	device = torch.device("cpu")
	pretrained = False
	n_branches = 3

	distortion_class_list = ["gaussian_blur", "gaussian_noise", "pristine"]

	b_mobilenet_pristine = B_MobileNet(n_classes, pretrained, n_branches, img_dim, exit_type, device)
	b_mobilenet_blur = B_MobileNet(n_classes, pretrained, n_branches, img_dim, exit_type, device)
	b_mobilenet_noise = B_MobileNet(n_classes, pretrained, n_branches, img_dim, exit_type, device)

	pristine_model = load_model(b_mobilenet_pristine, config.CLOUD_PRISTINE_MODEL_PATH, device)
	blur_model = load_model(b_mobilenet_blur, config.CLOUD_BLUR_MODEL_PATH, device)
	noise_model = load_model(b_mobilenet_noise, config.CLOUD_NOISE_MODEL_PATH, device)

	return [blur_model, noise_model, pristine_model], distortion_class_list


def select_distorted_model(pristineModel, blurModel, noiseModel, distortion_type):
	if ((distortion_type == 0) or (distortion_type == 1)):
		model = pristineModel
	elif(distortion_type == 2):
		model = noiseModel
	else:
		model = pristineModel

	return model


def read_temperature():
    df_pristine = pd.read_csv("./appCloud/api/services/models/temperature_calibration_pristine_b_mobilenet_21.csv")
    df_blur = pd.read_csv("./appCloud/api/services/models/temperature_calibration_gaussian_blur_b_mobilenet_21.csv")
    df_noise = pd.read_csv("./appCloud/api/services/models/temperature_calibration_gaussian_noise_b_mobilenet_21.csv")
    return [df_blur.iloc[0].values, df_noise.iloc[0].values, df_pristine.iloc[0].values]



class BranchesModelWithTemperature(nn.Module):
    def __init__(self, model, temperature):
        super(BranchesModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = temperature
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, conf_list, p_tar=0.5):
        return self.forwardEval(x, conf_list, p_tar)
    
    def forwardEval(self, x, conf_list, p_tar):

        output_list, conf_list, class_list  = [], [], []

        x = self.model.stages[-1](x)
        x = x.mean(3).mean(2)

        output = self.temperature_scale(self.model.fully_connected(x), -1)
        conf, infered_class = torch.max(self.softmax(output), 1)
    
        conf_list.append(conf.item())
        #class_list.append(infered_class)
        output_list.append(output)
    
        if (conf.item() >= p_tar):
           return output, conf.item(), infered_class
    
        else:
           max_conf = np.argmax(conf_list)
           return output, conf_list[max_conf], infered_class


    #def forwardEmulation(self, x, p_tar_list):


      
    def temperature_scale(self, logits, i):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        
        temperature = nn.Parameter(torch.from_numpy(np.array([self.temperature[i]]))).unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature


class NetworkConfiguration():
	def set_configuration(self, bandwidth, latency, distortion_lvl, robust):
		self.bandwidth = bandwidth
		self.latency = latency
		self.distortion_lvl = distortion_lvl
		self.robust = robust



