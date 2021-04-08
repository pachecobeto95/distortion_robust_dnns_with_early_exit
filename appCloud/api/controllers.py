from flask import Blueprint, g, render_template, request, jsonify, session, redirect, url_for, current_app as app
import json, os, time, sys, config
from .services import cloudProcessing


api = Blueprint("api", __name__, url_prefix="/api")


# Define url for the user send the image
@api.route('/cloud/cloudProcessing', methods=["POST"])
def cloud_receive_img():
	"""
	This function receives an image or feature map from edge device (Access Point)
	"""

	data_from_edge = request.json
	feature = data_from_edge['feature']
	start = data_from_edge['start']
	distortion_type = data_from_edge['distortion_type']
	conf_list = data_from_edge['conf']
	p_tar = data_from_edge['p_tar']
	distortion_lvl = data_from_edge['distortion_lvl']

	result = cloudProcessing.dnnInference(feature, conf_list, start, distortion_type, p_tar, distortion_lvl)
	return {"status": "ok"}

	if (result["status"] ==  "ok"):
		return jsonify(result), 200

	else:
		return jsonify(result), 500


@api.route('/cloud/cloudProcessingEmulation', methods=["POST"])
def cloud_receive_img_emulation():
	"""
	This function receives an image or feature map from edge device (Access Point)
	"""

	data_from_edge = request.json
	feature = data_from_edge['feature']
	start = data_from_edge['start']
	distortion_type = data_from_edge['distortion_type']
	conf_list = data_from_edge['conf']
	p_tar = data_from_edge['p_tar']
	distortion_lvl = data_from_edge['distortion_lvl']

	result = cloudProcessing.dnnInferenceEmulation(feature, conf_list, start, distortion_type, p_tar, distortion_lvl)
	return {"status": "ok"}

	if (result["status"] ==  "ok"):
		return jsonify(result), 200

	else:
		return jsonify(result), 500
