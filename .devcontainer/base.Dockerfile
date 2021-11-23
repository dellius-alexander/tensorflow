FROM dalexander2israel/tensorflow/tensorflow-object_detection_api-3:2.7.0-gpu
RUN conda install argcomplete && \
    eval "$(register-python-argcomplete conda)"

# setup alias for our new python environment
RUN alias python="/home/Tensorflow/anaconda3/envs/Tensorflow/bin/python3.8" && \
    echo 'alias python="/home/Tensorflow/anaconda3/envs/Tensorflow/bin/python3.8"' >> "${USER_HOME}/.bashrc" && \
    echo 'alias python="/home/Tensorflow/anaconda3/envs/Tensorflow/bin/python3.8"' >> "${USER_HOME}/.zshrc"
CMD [ "conda", "activate", "Tensorflow" ]