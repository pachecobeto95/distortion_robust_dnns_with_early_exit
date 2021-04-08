from flask import Flask, render_template, session, g
from appCloud.api.controllers import api     # import controlers => where thre is url to receive data
import torch
import torchvision.models as models
#from appCloud.api.services.utils import load_model
#from appCloud.api.services.cloudProcessing import decision_maker


app = Flask(__name__, static_folder="static")


#@app.before_first_request
#def before_first_request():
#	print("Execute Before Request")
#	decision_maker.making_decision()


app.config.from_object("config")
app.config['JSON_AS_ASCII'] = False
app.secret_key = 'xyz'
app.register_blueprint(api)
