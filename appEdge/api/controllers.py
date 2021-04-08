from flask import Blueprint, g, render_template, request, jsonify, session, redirect, url_for, current_app as app
import json, os, time, sys, config
#from .services.edgeProcessing import decision_maker, monitor_bandwidth
from .services import edgeProcessing
#from apscheduler.schedulers.background import BackgroundScheduler
import atexit, torch


api = Blueprint("api", __name__, url_prefix="/api")


# Define url for the user send the image
@api.route('/edge/recognition_cache', methods=["POST"])
def edge_receive_img():
	"""
	This function receives an image from user or client with smartphone or even a insurance camera 
	into smart sity context
	"""

	fileImg = request.files["img"]
	posted_data = json.load(request.files['data'])
	p_tar = posted_data["p_tar"]
	distortion_type = posted_data["distortion_type"]
	distortion_lvl = posted_data["distortion_lvl"]


	#Once reveived, the edge devices runs a DNN inference
	result = edgeProcessing.dnnInference(fileImg, p_tar, distortion_type, distortion_lvl)

	if (result["status"] ==  "ok"):
		return jsonify(result), 200

	else:
		return jsonify(result), 500



@api.route('/edge/edge_emulator', methods=["POST"])
def edgeEmulator():
	"""
	This function receives an image from user or client with smartphone or even a insurance camera 
	into smart sity context
	"""

	fileImg = request.files['img']
	posted_data = json.load(request.files['data'])

	p_tar = posted_data["p_tar"]
	nr_branch_2 = posted_data["nr_branch_2"] 
	nr_branch_3 = posted_data["nr_branch_3"]
	distortion_type = posted_data["distortion_type"]
	distortion_lvl = posted_data["distortion_lvl"]

	result = edgeProcessing.dnnInferenceEmulation(fileImg, p_tar, nr_branch_2, nr_branch_3, distortion_type, distortion_lvl)


	if (result["status"] ==  "ok"):
		return jsonify(result), 200

	else:
		return jsonify(result), 500


#This executes some function in background. The functions are making_decision that updates the partitioning layer 
#using optimization method and also monitor available upload bandwidth and record on a csv file. 
#scheduler = BackgroundScheduler()
#scheduler.add_job(func=decision_maker.making_decision, trigger="interval", seconds=config.DECISION_PERIOD)
#scheduler.add_job(func=monitor_bandwidth, trigger="interval", seconds=config.MONITOR_BANDWIDTH_PERIOD)
#scheduler.start()

# Shut down the scheduler when exiting the app
#atexit.register(lambda: scheduler.shutdown())
