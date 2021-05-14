# Early-exits Deep Neural Networks (DNNs) with expert branches

This repository contains all code developed for the paper "Early-exit deep neural networks for distorted images: providing an efficient edge offloading" by Roberto Pacheco and Rodrigo Couto. Moreover, this repository provides brief guides to help readers reproduce the codes and all the steps in the paper.

Before running the scripts in this repository, you should, first of all, create a virtual environment. Next, download the repository and install the necessary tools and libraries on your computer, including open-cv, pandas, numpy, pytorch, torchvision. To this end, run the requirements.txt file using the following command:

```
pip install -r requirements.txt
```

If you have any questions on this repository or the related paper, please feel free to send me an email. Email address: pacheco AT gta.ufrj.br. 

## Table of contents
* [Distorted Dataset](#distorted_dataset)


### Distorted Datasets

In this paper, we use the Caltech-256 dataset, which contains pristine (i.e., high-quality) images of common objects, such as motorbikes, airplanes, and shoes. This dataset has 257 classes with more than 80 instances in each. To evaluate the impact of image distortion on early-exits DNNs, we apply gaussian blur and gaussian noise into the images, using OpenCV and Numpy Python libraries. Hence, we generate a new Caltech-256 dataset, containing blurred and noisy images. For each distortion type, we consider five distortion levels.
We describe next these two distortion types.

#### Gaussian Blur
This image distortion occurs in an image when the camera is out of focus or in a foggy environment. The parameter ![equation](https://latex.codecogs.com/svg.image?\sigma_{GB}) defines the blur distortion level, so that a higher ![equation](https://latex.codecogs.com/svg.image?\sigma_{GB}) implies a more blurred image. This work uses ![equation](https://latex.codecogs.com/svg.image?\sigma_{GB}) from 1 to 5, with a step of 1. For each blur level, the kernel size is given by ![equation](https://latex.codecogs.com/svg.image?4\cdot\sigma_{GB}&plus;1). As mentioned before, we use the OpenCV library to apply blur distortion into the images from Caltech-256. 
The following image shows an example of a blurred image from the dataset with several blur levels.

