# Early-exits Deep Neural Networks (DNNs) with expert branches

This repository contains all code developed for the paper "Early-exit deep neural networks for distorted images: providing an efficient edge offloading" by Roberto Pacheco and Rodrigo Couto. Moreover, this repository provides brief guides to help readers reproduce the codes and all the steps in the paper. If you have any questions on this repository or the related paper, please feel free to send me an email. Email address: pacheco AT gta.ufrj.br. 


Before running the scripts in this repository, you should, first of all, create a virtual environment. Next, download the repository and install the necessary tools and libraries on your computer, including OpenCV, Pandas, Numpy, Pytorch, Torchvision. To this end, run the requirements.txt file using the following command:

```
pip install -r requirements.txt
```

## Table of contents
* [Distorted Datasets](#distorted-datasets)
* [Download Distorted Dataset](#download-distorted-dataset)
* [Generate Distorted Dataset](#generate-distorted-dataset)
* [Distortion Classifier](#distortion-classifier)
* [Early-exit DNNs with expert branches](#early-exit-dnns-with-expert-branches)
* [Experiments](#experiments)
* [Accuracy](#accuracy)
* [Offloading Probability](#offloading-probability)
* [End-to-end Latency](#end-to-end-latency)
* [Acknowledgments](#acknowledgments)

## Distorted Datasets

In this paper, we use the Caltech-256 dataset, which contains pristine (i.e., high-quality) images of common objects, such as motorbikes, airplanes, and shoes. This dataset has 257 classes with more than 80 instances in each. To evaluate the impact of image distortion on early-exits DNNs, we apply gaussian blur and gaussian noise into the images, using OpenCV and Numpy Python libraries. Hence, we generate a new Caltech-256 dataset, containing blurred and noisy images. For each distortion type, we consider five distortion levels. The [Section XX](#download-distorted-dataset) and [YY](#generate-distorted-dataset) presents the codes and procedures for downloading and generating the distorted datasets. 
We describe next these two distortion types.

#### Gaussian Blur
This image distortion occurs in an image when the camera is out of focus or in a foggy environment. The parameter ![equation](https://latex.codecogs.com/svg.image?\sigma_{GB}) defines the blur distortion level, so that a higher ![equation](https://latex.codecogs.com/svg.image?\sigma_{GB}) implies a more blurred image. This work uses ![equation](https://latex.codecogs.com/svg.image?\sigma_{GB}) from 1 to 5, with a step of 1. For each blur level, the kernel size is given by ![equation](https://latex.codecogs.com/svg.image?4\cdot\sigma_{GB}&plus;1). As mentioned before, we use the OpenCV library to apply blur distortion into the images from Caltech-256. 
The following image shows an example of a blurred image from the dataset with several blur levels.

![Blurred Image](https://github.com/pachecobeto95/distortion_robust_dnns_with_early_exit/blob/main/imgs_read_me/blur_monkey_levels.png)

#### Gaussian Noise
This distortion may appear in an image due to poor illumination conditions and the use of a low-quality image sensor. We use ![equation](https://latex.codecogs.com/svg.image?\sigma_{GN}) to define the noise distortion level, so that a higher ![equation](https://latex.codecogs.com/svg.image?\sigma_{GN}) implies a noisier image.
This work uses ![equation](https://latex.codecogs.com/png.image?\dpi{110}&space;\sigma_{GB}\in(1,&space;2,&space;3,&space;4,&space;5)). We use Numpy library to apply gaussian noise into the images. The following image shows an example of a noisy image from the dataset with several noise levels.

![Noisy Image](https://github.com/pachecobeto95/distortion_robust_dnns_with_early_exit/blob/main/imgs_read_me/noise_monkey_levels.png)

## Download Distorted Dataset

The original Caltech-256 dataset containing only pristine images can be download in the following link: http://www.vision.caltech.edu/Image_Datasets/Caltech256/256_ObjectCategories.tar

The blurred Caltech-256 can be downloaded in the following link: https://drive.google.com/drive/folders/1-3mB9mYqZ0ebzTdhTuF0VCil_Jm2OzH3?usp=sharing.

The noisy Caltech-256 can be downloaded in the following link: https://drive.google.com/drive/folders/1-1r8tPs70FFk7Q84pJYGH1qrgIhDBRF_?usp=sharing. 

For each distortion type, the distorted Caltech-256 dataset contains five different distortion level. 

## Generate Distorted Dataset

After downloading the original Caltech-256 dataset, we can generate the distorted dataset by running the Python script "distortionConverter.py", which can be downloaded via https://github.com/pachecobeto95/distortion_robust_dnns_with_early_exit/blob/main/distortionConverter.py. To generate the distorted datasets, follow the procedures below. 

1. Before running script "distortionConverter.py", make sure you have installed the required libraries, installing the "requirements.txt" as shown above. 

2. Create a directory called "dataset" and put the original Caltech-256 dataset into it. Next, create a directory "distorted_dataset" to storage the distorted dataset. To this end, run the following commands:   
```
mkdir dataset
unzip ./caltech256.zip -d ./dataset
mkdir ./dataset/distorted_dataset
```

3. Run the script "distortionConverter.py", as follows: ```python distortionConverter.py --distortion_type [DISTORTION_TYPE]```
   - distortion_type: describes the type of distortion you would like to generate. There are three options of distortion types available. 
     - "gaussian_blur" for blurred images;
     - "gaussian_noise" for noisy images;
     - "pristine" for pristine (i.e., high-quality) images.

4. The generated distorted dataset will be save in the "./dataset/distorted_dataset" directory. It is important to emphasize the the distortion levels of the generated distorted dataset are the same as those presented in the previous section and in the paper. However, you can change them in the script for your needs.

Another option for generating the distorted datasets is to run bash script: 
```
sudo bash generate_distorted_dataset.sh
``` 

## Distortion Classifier
This work uses the classifier proposed by Collabar, which can be accessed via https://github.com/CollabAR-Source/CollabAR-Code. This distortion classifier consists of a CNN (Convolutional Neural Network) trained to identify the predominant distortion contained in a given image. The distortion classifier is trained using a Fourier spectrum of the distorted images as the input data. 
Hence, for each distortion type, we generate a dataset containing the Fourier spectrum of the distorted images to train CNN to identify the predominant distortion in a given image. Once is trained, the distortion classifier classifies the image in one of the three distortion types: gaussian blur, gaussian noise or pristine, when no distortion is detected.

### Download the dataset for distortion classifier
The dataset used to train the distortion classifier can be in the following link: https://drive.google.com/drive/folders/1Kb8uZtYEvE-kJ2-dbNqHtpTmvK1bHVGT?usp=sharing. 

### Generate the dataset for distortion classifier
After downloading the original and distorted Caltech-256 dataset, we can generate the dataset to train the distortion classifier. To this end, you only need to provide the pristine Caltech-256 dataset, because the script can apply the distortion, convert to Fourier spectrum and save it. The distortion levels used to train the distortion classifier are the same as those presented in the paper and in the sections above. However, you can modify them in the script for your needs.

To generate the composite dataset of Fourier spectrum for distorted images used to train the distortion classifier, follow the procedures below. 
1. Before running the script, you have already installed the required libraries on your virtual environment.
2. Create a directory called "distortion_classifier_dataset" via: ``` mkdir ./dataset/distortion_classifier_dataset```.
3. Run the script as follows: ```python ./experiments/generate_distortion_classifier_dataset.py --distortion_type --root_path --save_path```.
   - distortion_type: indicates the type of distortion you would like to generate. There are three options of distortion types available.
     - "gaussian_blur" for blurred images;
     - "gaussian_noise" for noisy images;
     - "pristine" for pristine (i.e., high-quality) images.
   - root_path: indicates the path of pristine Caltech-256 dataset. If you have followed all the previous instructions correctly, the path of the Caltech-256 dataset is given by: ```./dataset/256_ObjectCategories```
   - save_path: indicates the path to save the generated dataset to train the distortion classifier. If you have follow the instruction 1,  the path is given by ```./dataset/distortion_classifier_dataset```

### Training the distortion classifier
We follow the procedures described by Collabar (https://github.com/CollabAR-Source/CollabAR-Code/blob/master/trainDisClassifer.py) to train the distortion classifier. Our training script is available in the following link: https://github.com/pachecobeto95/distortion_robust_dnns_with_early_exit/blob/main/experiments/training_distortion_classifier.py. The distortion classifier consists of a convolutional neural network (CNN) and its CNN model can be found in the file ```./experiments/distortionNet.py```. You need to provide the distorted datasets for training the distortion classifier model. If you follow the previous procedures, the path of the distorted datasets should be in ```./dataset/distortion_classifier_dataset```. It is important to notice that our training script train the distortion classifier models only on the distortion levels, described above and used in the paper. You can change them when you are performing the task of generating the distorted datasets according to your nedd. Then, you will train the distortion classifier model for this new distortion levels.

To train the distortion classifier, follow the procedure below:
1. Change the current directory to the experiments directory. 
2. Run the training script as follows: ```python training_distortion_classifier.py --dataset_path [DATASET_PATH] --model_path [MODEL_PATH]```
   - dataset_path: indicates the distortion datasets path.  
   - model_path: indicates the path to save the trained distortion classifier model.

After running the training, the script will save the trained distortion classifier model in the current directory (./experiments/distortionNetV2.pth). 
The trained distortion classifier model is available in: https://drive.google.com/file/d/1Ao1ikATJy-4bUrLsyUVy0JaRuVR5tc8f/view?usp=sharing

## Early-exit DNNs with expert branches

Our proposal is based on the fine-tuning strategy that trains a model using images with a particular distortion type. This work introduces the novel concept of expert branches, fine-tuning only the exits, which includes the side branches at edge and the DNN backbone's exit at the cloud. As shown in figure below, our proposal is composed of a distortion classifier and a early-exit DNN with expert branches, which works as follows. First, the edge receives an image that can be pristine, blurred or noisy from the end device. Next, the edge converts the received image into a Fourier spectrum and runs immediately the distortion classifier to identify the distortion type contained in the image. The distortion classifier model can classify the spectrum in one of the three distortion types: pristine, gaussian blur or gaussian noise. It is important to emphasize that if the distortion classifier detects no distortion, it means that the receives image is pristine. Based on the type of distortion identified by the classifier, the edge selects the most appropriate type of branch to be activated, i.e. those that have been fine-tuned specifically for the distortion type identified. 

This work uses the MobileNetV2 architecture as our DNN backbone. It is a lightweight model and optimized for running in resource-constrained hardware, such as edge devices. Next, we insert three early-exit points along the MobileNetV2, following the strategy proposed by SPINN. The training script used to fine-tune the early-exit DNNs with expert branches is available at link: https://github.com/pachecobeto95/distortion_robust_dnns_with_early_exit/blob/main/experiments/training_distortion_classifier.py

The trained early-exit DNN with expert branches can be downloaded in the following link: LINK

The model can be downloaded from the following link. Also, as an alternative, we have separately provided the early-exit DNNs with expert branches models for each distortion type, as follows:
* Pristine: https://drive.google.com/file/d/1-0wZhHDiuz2-VUNVSx0aZ3fPsgLdfP97/view?usp=sharing
* Gaussian Blur: https://drive.google.com/file/d/1-3b7hx3c40-lq3Z53C98V5w3fiFYfpgu/view?usp=sharing
* Gaussian Noise: https://drive.google.com/file/d/1CjRb0d6rLtK1nxsifQolmQhp1mX9I0N7/view?usp=sharing

The link of the blur expert model contains the early-exit DNN with branches expert in blurred images. Likewise, The link of the noise expert model contains the early-exit DNN with branches expert in noisy images.

![Early-exit DNN with expert branches](https://github.com/pachecobeto95/distortion_robust_dnns_with_early_exit/blob/main/imgs_read_me/robust_dnn_with_early_exit_alternative.png)

To fine-tune the early-exit DNN for each distortion type, follow the procedures below:
1. Change the current directory to the ```./experiment```
2. Run the training script, as follows: ```python training_distortion_classifier.py --distortion_type --root_path --dataset_path```
   - distortion_type: indicates the type of distortion you would like to fine-tune the branches, turning them into expert branches. 
   - root_path: indicates the root_path, in which the files will be saved. If you follow the previous procedures, the root_path parameter is ```./experiments```.
   - dataset_path: indicates the path to the original and pristine Caltech-256 dataset. If you follow the previous procedures, the dataset_path parameter is ```./dataset/256_ObjectCategories```.

This training scripts saves the trained early-expert DNNs with expert branches for each distortion type into directory defined by the parameter root_path, such as ```./experiments```.

## Experiments
At this stage, this section presents the experiments developed to evaluate our proposal -- early-exit DNN with expert branches -- considering the accuracy, offloading probability, and end-to-end latency. 

### Accuracy and Offloading Probability

### End-to-end Latency

This experiment evaluates the end-to-end latency, considering the early-exit DNN and the distortion classifier, where the edge works with the cloud to perform an image classification. The experiment emulates three components, representing the end device, the intermediate (edge) device, and the cloud server. 
The intermediate device and cloud server componentes are implemented using Python Flask framework (https://flask.palletsprojects.com/en/1.1.x).
The edge client and the cloud server communicate through HTTP.

We use a virtual network link between the edge server and the cloud, to emulate a communication through the Internet. We configure the network conditions on this link by using TC (http://lartc.org/manpages/tc.txt) and NETEM (https://wiki.linuxfoundation.org/networking/netem). Our experiments use different values of bandwidth and latency to analyze our proposal considering different cloud locations. The iPerf tool provides free servers geographically distributed around the world. The list of these free iPerf servers is available in the following link: https://iperf.fr/iperf-servers.php. We define the bandwidth values by measuring the throughput between our physical machine, in Rio de Janeiro (BR), and three iPerf servers. We use iPerf3 and ping to obtain, respectively, throughput and latency values.
The Table below shows the three chosen iPerf server and their bandwidth and latency (i.e., RTT) values.  

|   Server   |  City  |    Bandwidth    |  RTT  |
| :---:         |     :---:      |          :---: |     :---:   |
| speedtest.iveloz.net.br   | São Paulo (BR)     | 93.6 Mbps   |       10 ms  |
| iperf.scottlinux.com     | Fremont (US)       | 73.3 Mbps      |       180 ms    |
| bouygues.testdebit.info     | Paris (FR)       | 60.0 Mbps      |       216 ms    |

To obtain the throughput and latency values, follow the procedures below.
1. Use the iPerf3 tool to obtain the throughput value for each server in the Table, using the command:  ```iperf3 -c [SERVER] -p [PORT] -T [DURATION]```. For more information about this command, it is detailed in https://iperf.fr/iperf-doc.php
2. Use the ping tool to obtain the throughput value for each server in the Table, using the command:  ```ping [SERVER] -w [DURATION]```.

In this paper, we obtain the throughput and latency values for each server in the Table above. However, you can choose the iPerf server according to your needs. 

We can initialize the intermediate device and cloud server components just by running the bash script "run_robust_quality_early_dnn.sh" as follows: 
```
sudo bash run_robust_quality_early_dnn.sh
```
This bash script runs the following procedures:
1. Install Docker, following the instructions from https://docs.docker.com/engine/install/debian/ .
2. Download the wheels of OpenCV, Pytorch, Torchvision.
3. Build the docker image by running the dockerfile ```DockerfileEdge```.
4. Run the Docker container as follows: ```docker run -p 5000:5000 --cap-add=NET_ADMIN --cpus="1.0" -dit recog_container:apiEdge```
   - p: makes available a container's port to the host
   - cap-add provides Linux capabilities. In this case, ```--cap-add=NET_ADMIN``` provides networks capabilities. More specifically, this parameter enables us to use TC and Netem commands to emulate the network conditions between edge and cloud.   
   - cpus: limits the number of available cpu core. 
   - dit: runs the docker in background and in the interactive mode.

Since the intermediate container and the cloud component run in the same physical machine, we limit the number of CPU cores of the docker container, as shown before, and run the stress tool on it. It is important to emphasize that, when we run the basch script  bash run_robust_quality_early_dnn.sh, we have already installed the stress tool and run it, using the command: ```nohup stress --cpu 1 &``` 

To start running the end-to-end latency, run the Python script, as follows: ```nohup python distortedEndNode_exp.py --distortion_type [DISTORTION_TYPE]```. The parameter ```distortion_type``` indicates the type of distortion you would like to evaluate. There are three options of distortion types available:
* gaussian_blur
* gaussian_noise
* pristine


## Acknowledgments

The codes in this repository are developed by Roberto Pacheco. This work was done in laboratory of Grupo de Teleinformática e Automação at Universidade Federal do Rio de Janeiro (UFRJ).
This study was financed in part by the Coordenação de Aperfeiçoamento de Pessoal de Nível Superior - Brasil (CAPES). It was also supported by CNPq, FAPERJ Grants.

Contact for more informations:
* pacheco AT gta.ufrj.br





