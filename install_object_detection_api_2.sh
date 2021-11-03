#!/usr/bin/env bash
#####################################################################
# This script install the OBJECT DETECTION API on top of Tensorflow
## Prerequisite: you may need to install additional dependencies such as:
# apt-get update -y && apt-get install -y \
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
### USAGE: bash install_object_detection_api_2.sh [PARAM 1] [PARAM 1]\
######  
######  Parameters: ${1} = [PATH to Tensorflow Models Directory [models]]
######              ${2} = [Your OS flavor]
#####################################################################
# Instructions on how to build object detection model API
# https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md
# mkdir -p models_dir
# git clone https://github.com/tensorflow/models.git models_dir/
# Perspective of execution:=> From the root of the git repository ["models/"]
#### Type the Name of your Models Repo Directory
set -e  # Start execution from an isolated clean environment using "set -e".
MODELS_DIR_NAME=${1:-"Models"} # file discriptor $1
FLAVOR=${2:-"bionic"}  # file discriptor $2
MODELS_DIR=$(find ~- -type d -name "${MODELS_DIR_NAME}")
RESEARCH_DIR=${MODELS_DIR}/research
JSON_PARAM=("""
{\n
    'Models dir name': '${MODELS_DIR_NAME}',\n\
    'Models directory': '${MODELS_DIR}',\n\
    'Research directory': '${RESEARCH_DIR}',\n\
    'OS Flavor': '${FLAVOR}'\n\
}\n
""")
#####################################################################
#####################################################################
# Download Tensorflow Models Repo if it don't exist
if [ ! -d "${MODELS_DIR}" ]  && [ ! -h "${MODELS_DIR}" ]; then
    # create the models directory
    mkdir -p ${PWD}/${MODELS_DIR_NAME}/
    # clone the Models Repo
    git clone https://github.com/tensorflow/models.git  ${PWD}/${MODELS_DIR_NAME}/
    # Define some environment variables
    MODELS_DIR=$(find ~- -type d -name "${MODELS_DIR_NAME}")
    RESEARCH_DIR=${MODELS_DIR}/research
fi;
printf "${JSON_PARAM}"
#####################################################################
# Replace with your Linux OS Flavor, e.g. [OS=Flavor] ~ [ubuntu20.04=focal]
# sed -i 's/old-text/new-text/g' input.txt
# sed -i '1i\
# first_line_text
# '
DOCKERFILE="${MODELS_DIR}/research/object_detection/dockerfiles/tf2/Dockerfile"
echo $DOCKERFILE
# exit if Dockerfile not found
if [ ! -f "${DOCKERFILE}" ]; then
    printf "The Dockerfile does not exist please check your repo at: \n\t${DOCKERFILE}\n"
    exit 2
# check if build args are present in Dockerfile
elif [ -f "${DOCKERFILE}" ] && [[ $(cat $DOCKERFILE | grep -ic "ARG FLAVOR='${FLAVOR}'") -gt 0 ]]; then
    echo "Docker Build Args Set......"
    printf "\tARG FLAVOR='${FLAVOR}'\n"
# add build args to after "FROM" statement in Dockerfile if not present
elif [ -f "${DOCKERFILE}" ] && [[ $(cat $DOCKERFILE | grep -ic "ARG FLAVOR='${FLAVOR}'") -eq 0 ]]; then
    sed "/FROM tensorflow\/tensorflow/ a \
    ARG FLAVOR=\"${FLAVOR}\"
    " $DOCKERFILE | tee $DOCKERFILE
fi;
# now check for post-modified Dockerfile and update it if has not been modified
if [ -f "${DOCKERFILE}" ] && [[ $(cat $DOCKERFILE | grep -ic 'cloud-sdk-${FLAVOR}') -eq 0 ]]; then
    sed -i 's/$(lsb_release -c -s)/${FLAVOR}/g' $DOCKERFILE
    printf "\nModified Dockerfile:\n\t$(cat $DOCKERFILE | grep -i 'cloud-sdk-${FLAVOR}' | awk '{print $3}')\n"
else
    printf "Dockerfile has already been modified for the correct CLOUD-SDK-${FLAVOR}: \
        \n\t$(cat $DOCKERFILE | grep -i 'cloud-sdk-${FLAVOR}' | awk '{print $3}')\n"
fi;
# kill any duplicate running containers
CONTAINER=$(docker ps | grep -i 'object-detection' | awk '{print $1}')
if [ -z "${CONTAINER}" ]; then # check for running container
    # check for stopped containers
    CONTAINER=$(docker ps -a | grep -i 'object-detection' | awk '{print $1}')
    if [ ! -z "${CONTAINER}" ]; then
       docker rm "${CONTAINER}" # remove the stopped container
    fi;
    # no container then so nothing to do but move on

else # kill and remove the running container
    docker kill "${CONTAINER}" && docker rm "${CONTAINER}"
fi;
# Use the 'docker build' to build the Object Detection API
docker build --build-arg=FLAVOR=${FLAVOR} -t object-detection:0.1 \
    -f ${MODELS_DIR}/research/object_detection/dockerfiles/tf2/Dockerfile ${MODELS_DIR}
# Now RUN the Object Detection Image "od" you created above
docker run -it -v ${MODELS_DIR}:/home/tensorflow/models/research:rw -d --name object-detection object-detection:0.1
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
# Kill docker  object-detection container
docker stop object-detection && \
docker rm object-detection
# Test the installation.
python3 ${RESEARCH_DIR}/object_detection/builders/model_builder_tf2_test.py