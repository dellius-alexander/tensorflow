ARG TS_VERSION=2.6.1
FROM tensorflow/tensorflow:${TS_VERSION}-gpu
# Build arguments
ARG FLAVOR='bionic'
ARG DEBIAN_FRONTEND=noninteractive
ARG INSTALL_ZSH="true"
ARG USERNAME="tensorflow"
ARG USER_UID=1000
ARG USER_GID=${USER_UID}
ARG DEBIAN_FRONTEND=noninteractive
ARG CONDA_HOME="/home/tensorflow/anaconda3"
# Copy from original image to this image
COPY --from=dalexander2israel/tensorflow-object_detection_api:2.6.0 \
    /home/tensorflow/anaconda3 /home/tensorflow/anaconda3
# Define a few environmental variables
ENV CONDA_HOME="${CONDA_HOME}"
ENV PYTHONPATH="${CONDA_HOME}/envs/Tensorflow/bin/python3.8" 
