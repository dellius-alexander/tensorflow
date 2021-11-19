FROM registry.dellius.app/tensorflow/tensorflow-object_detection_api_01:2.6.2
RUN conda install argcomplete && \
    eval "$(register-python-argcomplete conda)"
CMD [ "conda", "activate", "Tensorflow" ]