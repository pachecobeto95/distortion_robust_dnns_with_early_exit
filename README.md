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

4. The generated distorted dataset will be save in the "./dataset/distorted_dataset" directory. It is important to emphasize the the distortion levels of the generated distorted dataset are the same as those presented in the previous section and in the paper. However, you can them in the script for your needs.

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
2. Create a directory called "distortion_classifier_dataset" via: ``` mkdir ./distortion_classifier_dataset```.
3. Run the script as follows: ```python ./experiments/generate_distortion_classifier_dataset.py --distortion_type --root_path --save_path```.
   - distortion_type: indicates the type of distortion you would like to generate. There are three options of distortion types available.
     - "gaussian_blur" for blurred images;
     - "gaussian_noise" for noisy images;
     - "pristine" for pristine (i.e., high-quality) images.
   - root_path: indicates the path of pristine Caltech-256 dataset. If you have followed all the previous instructions correctly, the path of the Caltech-256 dataset is given by: ```./dataset/256_ObjectCategories```
   - save_path: indicates the path to save the generated dataset to train the distortion classifier. If you have follow the instruction 1,  the path is given by ```./distortion_classifier_dataset```

### Training the distortion classifier


## Early-exit DNNs with expert branches

## Experiments
At this stage, this section presents the experiments developed to evaluate our proposal -- early-exit DNN with expert branches -- considering the accuracy, offloading probability, and end-to-end latency. 

### Accuracy

### Offloading Probability

### End-to-end Latency

## Acknowledgments

The codes in this repository are developed by Roberto Pacheco. This work was done in Grupo de Teleinformática e Automação at Universidade Federal do Rio de Janeiro (UFRJ).
This study was financed in part by the Coordenação de Aperfeiçoamento de Pessoal de Nível Superior - Brasil (CAPES). It was also supported by CNPq, FAPERJ Grants.

Contact for more informations:
* pacheco AT gta.ufrj.br





