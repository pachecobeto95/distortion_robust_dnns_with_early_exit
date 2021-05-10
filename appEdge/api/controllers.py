from flask import Blueprint, g, render_template, request, jsonify, session, redirect, url_for, current_app as app
import json, os, time, sys, config
#from .services.edgeProcessing import decision_maker, monitor_bandwidth
from .services import edgeProcessing
#from apscheduler.schedulers.background import BackgroundScheduler
import atexit, torch, requests
from .services.edgeProcessing import net_config

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
	distortion_type = posted_data["distortion_type"]
	distortion_lvl = posted_data["distortion_lvl"]
	robust = posted_data['robust']


	#Once reveived, the edge devices runs a DNN inference
	result = edgeProcessing.dnnInference(fileImg, distortion_type, distortion_lvl, robust)

	if (result["status"] ==  "ok"):
		return jsonify(result), 200

	else:
		return jsonify(result), 500


@api.route("/edge/starter_channel", methods=["POST"])
def starter_channel_edge():
	img = request.files["img"]
	result = sendToCloud(img)
	return jsonify(result), 200

def sendToCloud(img):
	files = {"img": img}
	r = requests.post(config.URL_CLOUD+"/api/cloud/starter_channel_cloud", files=files, timeout=10)
	if ((r.status_code != 200) or (r.status_code != 201)):
		result = {"status": "error"}
	else:
		result = {"status": "ok"}
	return result

@api.route("/edge/edge_update_network_config", methods=["POST"])
def updateNetwork():
	bandwidth = request.json["bandwidth"]
	latency = request.json["latency"]
	city = request.json["city"]
	net_config.set_configuration(bandwidth, latency, city)
	os.system("tc qdisc del dev eth0 root")
	os.system("tc qdisc add dev eth0 root netem delay %sms rate %smbit"%(latency, bandwidth))
	return jsonify({"status": "ok"}), 200

@api.route('/edge/edge_emulator', methods=["POST"])
def edgeEmulator():
	"""
	This function receives an image from user or client with smartphone or even a insurance camera 
	into smart sity context
	"""

	fileImg = request.files['img']
	posted_data = json.load(request.files['data'])

	nr_branch_2 = posted_data["nr_branch_2"] 
	nr_branch_3 = posted_data["nr_branch_3"]
	distortion_type = posted_data["distortion_type"]
	distortion_lvl = posted_data["distortion_lvl"]
	robust = posted_data["robust"]

	result = edgeProcessing.dnnInferenceEmulation(fileImg, nr_branch_2, nr_branch_3, distortion_type, distortion_lvl, robust)


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
