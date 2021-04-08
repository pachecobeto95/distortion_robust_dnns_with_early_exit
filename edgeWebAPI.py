from appEdge import app
import config

"""
iNITIALIZE EDGE API
"""
#from flask_script import Manager, Server


# configuring Host and Port from configuration files. 
app.debug = config.DEBUG
app.run(host=config.HOST_EDGE, port=config.PORT_EDGE)