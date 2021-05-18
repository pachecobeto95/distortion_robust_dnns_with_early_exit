#!/bin/bash
mkdir dataset
unzip ./caltech256.zip -d ./dataset
#tar -C ./dataset -xvf ./256_ObjectCategories.tar
mkdir ./dataset/distorted_dataset

python distortionConverter.py --distortion_type "pristine"
python distortionConverter.py --distortion_type "gaussian_blur"
python distortionConverter.py --distortion_type "gaussian_noise"

