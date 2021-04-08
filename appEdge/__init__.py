from flask import Flask, render_template, session, g
from appEdge.api.controllers import api     # import controlers => where thre is url to receive data
import torch
import torchvision.models as models
#from appEdge.api.services.utils import load_model
#from appEdge.api.services.edgeProcessing import decision_maker
#from flask_script import Manager, Server

app = Flask(__name__, static_folder="static")


@app.before_first_request
def before_first_request():
	print("Execute Before Request")


app.config.from_object("config")
app.config['JSON_AS_ASCII'] = False
app.secret_key = 'xyz'
app.register_blueprint(api)