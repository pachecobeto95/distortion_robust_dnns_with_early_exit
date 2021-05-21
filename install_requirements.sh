#!/bin/bash
apt update
apt install python3-pip
pip install Flask
pip install -r requirements.txt
pip install opencv-python
pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
