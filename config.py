import os

DIR_NAME = os.path.dirname(__file__)

DEBUG = True

#Period for the system run Decision Maker Module and Partitioning the DNN
DECISION_PERIOD = 30

# Edge Configuration 
HOST_EDGE = "127.0.0.1"
PORT_EDGE = 50001
URL_EDGE = "http://%s:%s"%(HOST_EDGE, PORT_EDGE)
SAVE_IMAGES_PATH_EDGE = os.path.join(DIR_NAME, "appEdge", "images")
SAVE_COMMUNICATION_TIME_PATH = os.path.join(DIR_NAME, "appEdge", "api", "communication_time", "communication_time_b_alexnet.csv")
PROCESSING_TIME_EDGE_PATH = os.path.join(DIR_NAME, "appEdge", "api", "processing_time", "processing_time_b_alexnet_cpu_edge_raspberry.csv")
PROCESSING_TIME_CLOUD_GPU_PATH = os.path.join(DIR_NAME, "appEdge", "api", "processing_time", "processing_time_b_alexnet_gpu_cloud.csv")
PROCESSING_TIME_CLOUD_CPU_PATH = os.path.join(DIR_NAME, "appEdge", "api", "processing_time", "processing_time_b_alexnet_cpu_cloud.csv")
OUTPUT_FEATURE_BYTES_SIZE = os.path.join(DIR_NAME, "appEdge", "api", "output_bytes_size", "output_feature_bytes_size_alexnet.csv")
NR_EARLY_EXITS_PATH = os.path.join(DIR_NAME, "appEdge", "api", "branchynet_data", "nr_early_exits_b_alexnet.csv")
DISTORTION_CLASSIFIER_MODEL_PATH = os.path.join(DIR_NAME, "appEdge", "api", "models", "distortionNet.pt")
SAVE_DISTORTED_IMG = os.path.join(DIR_NAME, "distorted_imgs")

#defining possible distortion levels
GAUSSIAN_BLUR_LIST = [1, 2, 3, 4, 5]
GAUSSIAN_NOISE_LIST = [5, 10, 20, 30, 40]

MONITOR_BANDWIDTH_PERIOD = 60

BRANCHES_POSITIONS = [2, 5, 7]

N_BRANCHES = 3

#Cloud Configuration 
HOST_CLOUD = "127.0.0.1"
PORT_CLOUD = 4000
URL_CLOUD = "http://%s:%s"%(HOST_CLOUD, PORT_CLOUD)
SAVE_IMAGES_PATH_CLOUD = os.path.join(DIR_NAME, "appCloud", "images")




#Model Paths of CLOUD MODEL
CLOUD_PRISTINE_MODEL_PATH = os.path.join(".", "appCloud", "api", "services", "models", "pristine_model_b_mobilenet_caltech_17.pth")
CLOUD_BLUR_MODEL_PATH = os.path.join(".", "appCloud", "api", "services", "models", "gaussian_blur_distorted_model_mobilenet_caltech_21_freezing_3.pth")
CLOUD_NOISE_MODEL_PATH = os.path.join(".", "appCloud", "api", "services", "models", "gaussian_noise_distorted_model_mobilenet_caltech_21_freezing_3.pth")


#Inference Time Result Paths of CLOUD MODEL
RESULTS_INFERENCE_TIME_CLOUD = os.path.join(DIR_NAME, "appCloud", "api", "services", "result")

#Model Paths of EDGE MODEL
EDGE_PRISTINE_MODEL_PATH = os.path.join(".", "appEdge", "api", "services", "models", "pristine_model_b_mobilenet_caltech_17.pth")
EDGE_BLUR_MODEL_PATH = os.path.join(".", "appEdge", "api", "services", "models", "gaussian_blur_distorted_model_mobilenet_caltech_21_freezing_3.pth")
EDGE_NOISE_MODEL_PATH = os.path.join(".", "appEdge", "api", "services", "models", "gaussian_noise_distorted_model_mobilenet_caltech_21_freezing_3.pth")

#Inference Time Result Paths of EDGE MODEL
RESULTS_INFERENCE_TIME_EDGE = os.path.join(DIR_NAME, "appEdge", "api", "services", "result")

save_idx_path = os.path.join(DIR_NAME, "dataset", "save_idx_b_mobilenet_caltech_21.npy")
dataset_path = os.path.join(DIR_NAME, "dataset", "256_ObjectCategories", "256_ObjectCategories")


