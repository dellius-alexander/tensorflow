FROM dalexander2israel/tensorflow-object_detection_api:2.6.0
ARG CONDA_HOME="/home/tensorflow/anaconda3"
RUN rm -rf /tmp/conda-tmp  /tmp/library-scripts  /home/tensorflow/models && \
    apt-get autoremove -y && apt-get clean
# # Add new user to avoid running as root
# RUN useradd -ms /bin/bash tensorflow
# USER tensorflow
# WORKDIR /home/tensorflow
# Define a few environmental variables
ENV CONDA_HOME="/home/tensorflow/anaconda3"
ENV PYTHONPATH="/home/tensorflow/anaconda3/envs/Tensorflow/bin/python3.8" 
ENV CONDAPATH="/home/tensorflow/anaconda3/condabin/"
ENV PATH="${CONDAPATH}:${PATH}"