#!/bin/bash

apt update
apt install python3-pip
pip install -r requirements.txt
pip install kaggle
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

#apt-get update
#apt-get install apt-transport-https
#apt-get install ca-certificates
#apt-get install curl
#apt-get install gnupg
#apt-get install lsb-release
#curl -fsSL https://download.docker.com/linux/debian/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo \
#  "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/debian \
#  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

#sudo apt-get update
#sudo apt-get install docker-ce docker-ce-cli containerd.io
pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
wget https://files.pythonhosted.org/packages/2a/9a/ff309b530ac1b029bfdb9af3a95eaff0f5f45f6a2dbe37b3454ae8412f4c/opencv_python-4.5.1.48-cp38-cp38-manylinux2014_x86_64.whl
wget https://download.pytorch.org/whl/cu101/torch-1.7.1%2Bcu101-cp38-cp38-linux_x86_64.whl
wget https://download.pytorch.org/whl/cu101/torchvision-0.8.2%2Bcu101-cp38-cp38-linux_x86_64.whl
echo "Starting Edge Container"
docker build -f DockerfileEdge -t recog_container:apiEdge10 .
docker run -p 5000:5000 --cap-add=NET_ADMIN -dit recog_container:apiEdge10
echo "Completed Edge Container"
sleep 5
echo "Starting Cloud Server"
nohup python cloudWebAPI.py > logCloudWebAPI.txt &
echo "Completed Cloud Server"
#sleep 30
#echo "Executing Edge Node"
#nohup python distortedEndNode_exp.py --distortion_type "gaussian_blur" > logBlur.txt &
