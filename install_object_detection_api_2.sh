#!/usr/bin/env bash
#####################################################################
# This script install the OBJECT DETECTION API on top of Tensorflow
## Prerequisite: you may need to install additional dependencies such as:
# git \
# gpg-agent \
# python3-cairocffi \
# protobuf-compiler \
# python3-pil \
# python3-lxml \
# python3-tk \
# wget \
# gcc \
# build-essentials
#####################################################################
### USAGE: bash install_object_detection_api_2.sh [Enter the Models Directory Repo Path you Downloaded Above [models_dir]]
#####################################################################
# Instructions on how to build object detection model API
# https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md
# mkdir -p models_dir
# git clone https://github.com/tensorflow/models.git models_dir/
# Perspective of execution:=> From the root of the git repository [models_dir/]
#### Type the Name of your Models Repo Directory
set -e
MODELS_DIR_NAME=${1:-"Models"}
MODELS_DIR=$(find ~- -type d -name "${MODELS_DIR_NAME}")
RESEARCH_DIR=${MODELS_DIR}/research
printf """\n
{\n
\t'Models dir name': '${MODELS_DIR_NAME}',\n \
\t'Models directory': '${MODELS_DIR}',\n \
\t'Research directory': '${RESEARCH_DIR}'\n \
}\n
"""

# Download Tensorflow Models Repo if it don't exist
if [ ! -d "${MODELS_DIR}" ]  && [ ! -h "${MODELS_DIR}" ]; then
    mkdir -p ${MODELS_DIR_NAME}/
    git clone https://github.com/tensorflow/models.git  ${MODELS_DIR_NAME}/
    MODELS_DIR=$(find ~- -type d -name "${MODELS_DIR_NAME}")
    RESEARCH_DIR=${MODELS_DIR}/research
    printf """\n
    {\n
    \t'Models dir name': '${MODELS_DIR_NAME}',\n \
    \t'Models directory': '${MODELS_DIR}',\n \
    \t'Research directory': '${RESEARCH_DIR}'\n \
    }\n
    """
fi;
# sed -i 's/old-text/new-text/g' input.txt
sed -i 's/$(lsb_release -c -s)/bionic/g' Models/research/object_detection/dockerfiles/tf2/Dockerfile
# # exit 0
# cat >>${MODELS_DIR}/research/object_detection/dockerfiles/tf2/Dockerfile<<EOF
# \n
# ARG MODELS_DIR_NAME="Models"
# ARG MODELS_DIR="/home/tensorflow/models"
# ARG RESEARCH_DIR="/home/tensorflow/models/research"
# EOF
# Build the docker container to run the Object Detection API
docker build -f ${MODELS_DIR}/research/object_detection/dockerfiles/tf2/Dockerfile -t od ${MODELS_DIR}
# Now RUN the Object Detection Image "od" you created above
docker run -it -v ${MODELS_DIR}:/home/tensorflow/models/research:rw -d od
# change directory and compile Object Detection API protocols
cd ${RESEARCH_DIR}
# Compile the Object Detection API protocols into --python_out=<some directory | [default]. >
protoc object_detection/protos/*.proto --python_out=${RESEARCH_DIR}/object_detection/protos
# Install Object Detection into TensorFlow Object Detection API.
cp ${MODELS_DIR}/research/object_detection/packages/tf2/setup.py ${RESEARCH_DIR}
# # Because Tensorflow-gpu 2.2.0 depends on gast==0.3.3, h5py<2.11.0,>=2.10.0, tensorboard<2.3.0,>=2.2.0
# ## so we have to uninstall and reinstall these packages
# pip uninstall gast h5py tensorboard
# pip install gast==0.3.3 h5py<2.11.0,>=2.10.0 tensorboard<2.3.0,>=2.2.0
# install the built module into your python/conda environment 
python3 -m pip install --use-feature=2020-resolver ${RESEARCH_DIR}
# Test the installation.
python3 ${RESEARCH_DIR}/object_detection/builders/model_builder_tf2_test.py
