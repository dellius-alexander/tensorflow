# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.202.5/containers/python-3-anaconda/.devcontainer/base.Dockerfile
ARG TFVERSION=2.6.1
FROM tensorflow/tensorflow:${TFVERSION}-gpu as setup_gcsdk

ARG NODE_VERSION="true"
ARG INSTALL_ZSH="true"
ARG USERNAME="tensorflow"
ARG USER_UID=1000
ARG USER_GID=${USER_UID}
ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHONVERSION=3.8
ARG WORK_DIR="/home/tensorflow/Tensorflow"
ARG CONDA_HOME="/home/tensorflow/anaconda3"

COPY library-scripts/* /tmp/library-scripts/
RUN ls -lia /tmp/library-scripts
# RUN if [[ -f  "/tmp/library-scripts/common-debian.sh"  ]]; then \
#         cat "/tmp/library-scripts/common-debian.sh"; else exit 1; fi;
# # Setup user environment
RUN apt-get update \
    && /bin/bash /tmp/library-scripts/common-debian.sh "${INSTALL_ZSH}" "${USERNAME}" "${USER_UID}" "${USER_GID}" "true" "true" "true"

# Install apt dependencies
RUN apt-get update && apt-get install -y \
    git \
    gpg-agent \
    python3-cairocffi \
    protobuf-compiler \
    python3-pil \
    python3-lxml \
    python3-tk \
    python3-opencv \
    zsh \
    wget

# Installs google cloud sdk, this is mostly for using gsutil to export model.
RUN wget -nv \
    https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz && \
    mkdir /root/tools && \
    tar xvzf google-cloud-sdk.tar.gz -C /root/tools && \
    rm google-cloud-sdk.tar.gz && \
    /root/tools/google-cloud-sdk/install.sh --usage-reporting=false \
        --path-update=false --bash-completion=false \
        --disable-installation-options && \
    rm -rf /root/.config/* && \
    ln -s /root/.config /config && \
    rm -rf /root/tools/google-cloud-sdk/.install/.backup
# Path configuration
ENV PATH $PATH:/root/tools/google-cloud-sdk/bin
# Setup anaconda
FROM setup_gcsdk as setup_conda
# Install oh-my-zsh
RUN rm -rf ~/.oh-my-zsh && sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "Y"
# Copy environment.yml (if found) to a temp location so we update the environment. Also
# copy "noop.txt" so the COPY instruction does not fail if no environment.yml exists.
COPY environment.yml* /tmp/conda-tmp/
# Install anaconda into our container USER dev environment
# Usage:  ./anaconda.sh [[0][-b {LICENSE MOFE flag}]], 
#                       [[1][-p {PREFIX flag}]], 
#                       [[2][PATH to ${CONDA_HOME}]], 
#                       [[3][PATH to your app directory]], 
#                       [[4][USERNAME]],
#                       [[5][ENV_FILE]]
RUN /bin/bash /tmp/library-scripts/anaconda.sh \
    "-b" \
    "-p" \
    "${CONDA_HOME}" \
    "${WORK_DIR}" \
    "${USERNAME}" \
    "/tmp/conda-tmp/environment.yml"
# setup alias for our new python environment
RUN alias python='/usr/bin/python3.4' && \
    echo 'alias python="/usr/bin/python3.4"' >> /home/tensorflow/.bashrc && \
    echo 'alias python="/usr/bin/python3.4"' >> /home/tensorflow/.zshrc
# Define our work directory
WORKDIR /${WORK_DIR}
# copy a symbolic link of conda to our .local/bin directory
# RUN ln -s "${CONDAPATH}/conda"  "/home/tensorflow/.local/bin"
# Export a few changes to path variables and environment variables
ENV PATH="/home/tensorflow/.local/bin:${CONDAPATH}:${CONDA_HOME}/envs/Tensorflow/bin:${PATH}"
ENV CONDA_HOME="${CONDA_HOME}"
ENV PYTHONPATH="${CONDA_HOME}/envs/Tensorflow/bin/python${PYTHONVERSION}" 
ENV CONDAPATH="${CONDA_HOME}/condabin/"
# Make sure gsutil will use the default service account
RUN echo '[GoogleCompute]\nservice_account = default' > /etc/boto.cfg

# Setup object detection on top of Tenserflow
FROM setup_conda
RUN cd /home/tensorflow  && \
    # conda initialize
    source "/home/tensorflow/anaconda3/etc/profile.d/conda.sh" && \
    conda activate Tensorflow
## Copy this code (make sure you are under the ../models/research directory)
RUN git clone https://github.com/tensorflow/models.git /home/tensorflow/models && \
    ls -lia /home/tensorflow/models/research

# Compile protobuf configs
RUN cd /home/tensorflow/models/research && \
    protoc object_detection/protos/*.proto  \
    --python_out=.
WORKDIR /home/tensorflow/Tensorflow/

RUN cp /home/tensorflow/models/research/object_detection/packages/tf2/setup.py /home/tensorflow/models/research

RUN python -m pip install -U pip
RUN python -m pip install /home/tensorflow/models/research

# Clean up
RUN apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*  /tmp/library-scripts/ /tmp/conda-tmp/ /home/tensorflow/models/

# ENTRYPOINT ["python", "object_detection/model_main_tf2.py"]
ENTRYPOINT [ "conda", "activate", "Tensorflow" ]