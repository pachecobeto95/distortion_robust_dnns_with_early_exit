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
	device = 'cpu'
	pretrained = False
	n_branches = 3
	distortion_classes = ["gaussian_blur", "gaussian_blur", "gaussian_noise", "pristine"]


	b_mobilenet_pristine = B_MobileNet(n_classes, pretrained, n_branches, img_dim, exit_type, device)
	b_mobilenet_blur = B_MobileNet(n_classes, pretrained, n_branches, img_dim, exit_type, device)
	b_mobilenet_noise = B_MobileNet(n_classes, pretrained, n_branches, img_dim, exit_type, device)

	pristine_model = load_model(b_mobilenet_pristine, config.EDGE_PRISTINE_MODEL_PATH, device)
	blur_model = load_model(b_mobilenet_blur, config.EDGE_BLUR_MODEL_PATH, device)
	noise_model = load_model(b_mobilenet_noise, config.EDGE_NOISE_MODEL_PATH, device)

	return [blur_model, blur_model, noise_model, pristine_model], distortion_classes

def read_temperature():
	df_pristine = pd.read_csv("./appEdge/api/services/models/temperature_calibration_pristine_b_mobilenet_21.csv")
	df_blur = pd.read_csv("./appEdge/api/services/models/temperature_calibration_gaussian_blur_b_mobilenet_21.csv")
	df_noise = pd.read_csv("./appEdge/api/services/models/temperature_calibration_gaussian_noise_b_mobilenet_21.csv")
	return [df_blur.iloc[0].values, df_blur.iloc[0].values, df_noise.iloc[0].values, df_pristine.iloc[0].values]

class DistortionNet(nn.Module):
	def __init__(self):
		super(DistortionNet, self).__init__()
		self.features = nn.Sequential(nn.Conv2d(3, 32, kernel_size=3, stride=2),
			nn.Conv2d(32, 16, kernel_size=3, stride=1),
			nn.MaxPool2d(2, stride=2),
			nn.Conv2d(16, 1, kernel_size=1, stride=1),
			nn.MaxPool2d(2, stride=2))

		self.classifier = nn.Linear(729, 4)

	def forward(self, x):
		x = self.features(x)
		x = x.view(x.size(0), -1)
		x = self.classifier(x)

		return x


def loadDistortionClassifier():
	distortion_classifier = DistortionNet()
	distortion_classifier.load_state_dict(torch.load(config.DISTORTION_CLASSIFIER_MODEL_PATH, map_location=torch.device('cpu'))["model_state_dict"])
	
	return distortion_classifier



def apply_fourier_transformation(img):
	gray_img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
	dft = np.fft.fft2(gray_img)
	fshift = np.fft.fftshift(dft)
	magnitude_spectrum = 20*np.log(np.abs(fshift))
	result = Image.fromarray((magnitude_spectrum).astype(np.uint8)).convert("RGB")
	result = transforms.ToTensor()(result)
	return torch.stack([result])


def inferenceTransformation(img):

	mean = [0.457342265910642, 0.4387686270106377, 0.4073427106250871]
	std = [0.26753769276329037, 0.2638145880487105, 0.2776826934044154]

	data_transformation = transforms.Compose([
		transforms.ToPILImage(),
		transforms.Resize(330), 
		transforms.CenterCrop(300),
		transforms.ToTensor(),
		transforms.Normalize(mean=mean, std=std)])

	
	return data_transformation(img[0]).unsqueeze(0)


def inferenceTransformationDistortionClassifier(img_numpy):

	data_transformation = transforms.Compose([
		transforms.ToPILImage(),
		transforms.Resize(224), 
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.ToPILImage()])

	img_pil = data_transformation(img_numpy)
	img_fft = apply_fourier_transformation(img_pil)


	return img_fft.to("cpu")

def select_distorted_model(pristineModel, blurModel, noiseModel, distortion_type, pristine_temp, blur_temp, noise_temp):
	if ((distortion_type.item() == 0) or (distortion_type.item() == 1)):
		model = blurModel
		temperature = blur_temp

	elif(distortion_type.item() == 2):
		model = noiseModel
		temperature = noise_temp

	else:
		model = pristineModel
		temperature = pristine_temp

	
	temperature_list = [1.2679497003555298,0.9083033204078674,0.6282054781913757,0.9926857352256775]
	return model, temperature_list



class BranchesModelWithTemperature(nn.Module):
    def __init__(self, model, temperature):
        super(BranchesModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = temperature
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, p_tar=0.5, train=True, repo=False):
        return self.forwardEval(x, p_tar)
    
    def forwardEval(self, x, p_tar):
      output_list, conf_list, class_list  = [], [], []
      for i, exitBlock in enumerate(self.model.exits):
        x = self.model.stages[i](x)
        output_branch = exitBlock(x)
        output_branch =  self.temperature_scale(output_branch, i)
        conf, infered_class = torch.max(self.softmax(output_branch), 1)

        if (conf.item() > p_tar):
          return output_branch, conf.item(), infered_class

        else:
          output_list.append(output_branch)
          conf_list.append(conf.item())
          class_list.append(infered_class)

      return x, conf_list, None

    def forwardEmulation(self, x, p_tar_list):
      output_list, conf_list, class_list  = [], [], []
      for i, exitBlock in enumerate(self.model.exits):
        x = self.model.stages[i](x)
        output_branch = exitBlock(x)
        output_branch =  self.temperature_scale(output_branch, i)
        conf, infered_class = torch.max(self.softmax(output_branch), 1)

        if (conf.item() > p_tar_list[i]):
          return output_branch, conf, infered_class

        else:
          output_list.append(output_branch)
          conf_list.append(conf.item())
          class_list.append(infered_class)

      return x, conf_list, None


      
    def temperature_scale(self, logits, i):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        
        temperature = nn.Parameter(torch.from_numpy(np.array([self.temperature[i]]))).unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature


def compute_p_tar_list(nr_samples_edge, nr_branch_2, nr_branch_3):
	if (nr_samples_edge < nr_branch_2):
		p_tar_list = [1, 0, 0, 0]
	elif ( (nr_samples_edge >= nr_branch_2) and (nr_samples_edge < nr_branch_3)):
		p_tar_list = [1, 1, 0, 0]
	else:
		p_tar_list = [1, 1, 1, 0]

	return p_tar_list


