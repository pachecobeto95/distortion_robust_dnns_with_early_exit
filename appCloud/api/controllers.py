from flask import Blueprint, g, render_template, request, jsonify, session, redirect, url_for, current_app as app
import json, os, time, sys, config
from .services import cloudProcessing
from .services.cloudProcessing import net_config

api = Blueprint("api", __name__, url_prefix="/api")


# Define url for the user send the image
@api.route('/cloud/cloudProcessing', methods=["POST"])
def cloud_receive_img():
	"""
	This function receives an image or feature map from edge device (Access Point)
	"""

	data_from_edge = request.json
	#feature = data_from_edge['feature']
	#start = data_from_edge['start']
	#distortion_type = data_from_edge['distortion_type']
	#conf_list = data_from_edge['conf']
	#p_tar = data_from_edge['p_tar']
	#distortion_lvl = data_from_edge['distortion_lvl']

	result = cloudProcessing.dnnInference(data_from_edge["feature"], data_from_edge["conf_list"], data_from_edge["distortion_type"])

	if (result["status"] ==  "ok"):
		return jsonify(result), 200

	else:
		return jsonify(result), 500

@api.route('/cloud/starter_channel_cloud', methods=["POST"])
def starter_channel_cloud():
	data = request.files["img"]
	return jsonify({"status": "ok"}), 200

@api.route('/cloud/cloud_update_network_config', methods=["POST"])
def update_network_config():
	data_from_edge = request.json
	latency = data_from_edge["latency"]
	bandwidth = data_from_edge["bandwidth"]
	os.system("tc qdisc del dev docker0 root")
	os.system("tc qdisc add dev docker0 root netem delay %sms rate %smbit"%(latency, bandwidth))
	return jsonify({"status": "ok"}), 200


@api.route('/cloud/cloudProcessingEmulation', methods=["POST"])
def cloud_receive_img_emulation():
	"""
	This function receives an image or feature map from edge device (Access Point)
	"""

	data_from_edge = request.json
	result = cloudProcessing.dnnInferenceEmulation(data_from_edge["feature"], data_from_edge["conf"], data_from_edge["distortion_type"])

	if (result["status"] ==  "ok"):
		return jsonify(result), 200

	else:
		return jsonify(result), 500
