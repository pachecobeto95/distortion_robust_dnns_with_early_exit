from flask import jsonify, session, current_app as app
import os, pickle, requests, sys, time
import numpy as np, json
import torchvision.models as models
import torch
import cv2
import datetime, io
import torchvision.transforms as transforms
from torchvision import datasets, transforms
from PIL import Image
from mobilenet import B_MobileNet


def transform_image(image_bytes):
	my_transforms = transforms.Compose([transforms.Resize(330),
		transforms.CenterCrop(300),
		transforms.ToTensor(),
		transforms.Normalize([0.457342265910642, 0.4387686270106377, 0.4073427106250871], [0.26753769276329037, 0.2638145880487105, 0.2776826934044154])])

	image = Image.open(io.BytesIO(image_bytes))
	return my_transforms(image).unsqueeze(0)


branchynet = B_MobileNet(258, True, 3, 300, None, 'cpu')
branchynet.load_state_dict(torch.load("./models/pristine_model_b_mobilenet_caltech_17.pth", map_location=torch.device('cpu'))["model_state_dict"])
branchynet = branchynet.to('cpu')

with open(os.path.join(".", "001_0001.jpg"), "rb") as f:
	image_bytes = f.read()
	tensor = transform_image(image_bytes)
	#branchynet = B_MobileNet(258, True, 3, 300, None, 'cpu')
	#branchynet.load_state_dict(torch.load("./models/pristine_model_b_mobilenet_caltech_17.pth", map_location=torch.device('cpu'))["model_state_dict"])
	#branchynet = branchynet.to('cpu')
	torch.backends.cudnn.enabled = False
	branchynet.eval()
	with torch.no_grad():
		print(branchynet(tensor))
