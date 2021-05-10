from flask import jsonify, session, current_app as app
import os, pickle, requests, sys, time
import numpy as np, json
import torchvision.models as models
import torch
import datetime
import time, io
import pandas as pd
import torchvision.transforms as transforms
from PIL import Image
from mobilenet import B_MobileNet



def transform_image(image_bytes):
	my_transforms = transforms.Compose([transforms.Resize(330),
		transforms.CenterCrop(300),
		transforms.ToTensor(),
		transforms.Normalize([0.457342265910642, 0.4387686270106377, 0.4073427106250871], [0.26753769276329037, 0.2638145880487105, 0.2776826934044154])])

	image = Image.open(io.BytesIO(image_bytes))
	return my_transforms(image).unsqueeze(0)




model = B_MobileNet(258, False, 3, 300, None, 'cpu')
model.load_state_dict(torch.load("./models/pristine_model_b_mobilenet_caltech_17.pth", map_location="cpu")["model_state_dict"])

with open("001_0001.jpg", "rb") as f:
	x = transform_image(f.read())
	model.eval()
	with torch.no_grad():
		print(model(x))
