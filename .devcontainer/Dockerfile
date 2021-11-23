# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.202.5/containers/python-3-anaconda/.devcontainer/base.Dockerfile
ARG TFVERSION="2.7.0"
FROM tensorflow/tensorflow:${TFVERSION}-gpu as setup_gcsdk

ARG NODE_VERSION="true"
ARG INSTALL_ZSH="true"
ARG USERNAME="Tensorflow"
ARG USER_UID=1000
ARG USER_GID=${USER_UID}
ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHONVERSION=3.8
ARG USER_HOME="/home/${USERNAME}"
ARG WORK_DIR="/home/${USERNAME}/Tensorflow"
ARG CONDA_HOME="/home/${USERNAME}/anaconda3"

COPY library-scripts/* /tmp/library-scripts/
RUN ls -lia /tmp/library-scripts
# RUN if [[ -f  "/tmp/library-scripts/common-debian.sh"  ]]; then \
#         cat "/tmp/library-scripts/common-debian.sh"; else exit 1; fi;
# # Setup user environment
RUN apt-get update \
    && /bin/bash /tmp/library-scripts/common-debian.sh \
    "${INSTALL_ZSH}" "${USERNAME}" "${USER_UID}" "${USER_GID}" "true" "true" "true"

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
USER ${USERNAME}
# Installs google cloud sdk, this is mostly for using gsutil to export model.
RUN curl -fsSL \
    https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz \
    -o /tmp/google-cloud-sdk.tar.gz && \
    mkdir "${USER_HOME}/tools" && \
    tar xvzf /tmp/google-cloud-sdk.tar.gz -C "/${USER_HOME}/tools" && \
    rm /tmp/google-cloud-sdk.tar.gz && \
    "${USER_HOME}/tools/google-cloud-sdk/install.sh" \
        --usage-reporting=false \
        --path-update=false \
        --bash-completion=false \
        --disable-installation-options
RUN    rm -rf "${USER_HOME}/.config/*"
# RUN ls /
# RUN    CONFIG=$(ls / | grep -i 'config') &2>/dev/null
# RUN    if [[ -z ${CONFIG} ]]; then CONFIG=$(ls ${USER_HOME} | grep -i 'config' &2>/dev/null) fi; 
RUN    ln -s "${USER_HOME}/.config" "${USER_HOME}/config" && \
    rm -rf "${USER_HOME}/tools/google-cloud-sdk/.install/.backup"
# Path configuration
ENV PATH="$PATH:${USER_HOME}/tools/google-cloud-sdk/bin"
# Setup anaconda
FROM setup_gcsdk as setup_conda
# Install oh-my-zsh
RUN rm -rf ~/.oh-my-zsh && sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "Y"
# Copy environment.yml (if found) to a temp location so we update the environment. Also
# copy "noop.txt" so the COPY instruction does not fail if no environment.yml exists.
COPY environment.yml* /tmp/conda-tmp/
USER root
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
RUN alias python='/usr/bin/python3.8' && \
    echo 'alias python="/usr/bin/python3.8"' >> "${USER_HOME}/.bashrc" && \
    echo 'alias python="/usr/bin/python3.8"' >> "${USER_HOME}/.zshrc"
# Define our work directory
WORKDIR ${WORK_DIR}
# copy a symbolic link of conda to our .local/bin directory
# RUN ln -s "${CONDAPATH}/conda"  "/home/tensorflow/.local/bin"
# Export a few changes to path variables and environment variables
ENV CONDAPATH="${CONDA_HOME}/condabin/"
ENV PATH="${USER_HOME}/.local/bin:${CONDAPATH}:${CONDA_HOME}/envs/Tensorflow/bin:${PATH}"
ENV CONDA_HOME="${CONDA_HOME}"
ENV PYTHONPATH="${CONDA_HOME}/envs/Tensorflow/bin/python${PYTHONVERSION}" 

# Make sure gsutil will use the default service account
RUN echo '[GoogleCompute]\nservice_account = default' > /etc/boto.cfg

# Setup object detection on top of Tenserflow
FROM setup_conda
RUN cd "${USER_HOME}"  && \
    # conda initialize
    source "${CONDA_HOME}/etc/profile.d/conda.sh" && \
    conda activate Tensorflow
# Define a work directory
WORKDIR /home/tensorflow/
## Copy this code (make sure you are under the ../models/research directory)
RUN git clone https://github.com/tensorflow/models.git "${WORK_DIR}/models" && \
    ls -lia "${WORK_DIR}/models/research"

RUN python3 -m pip install -U pip
# Compile protobuf configs
RUN cd "${WORK_DIR}/models/research" && \
    protoc object_detection/protos/*.proto  \
    --python_out=.
# copy setup.py to models/research
RUN cp "${WORK_DIR}/models/research/object_detection/packages/tf2/setup.py" "${WORK_DIR}/models/research"
RUN ls -lia "${WORK_DIR}/models"
RUN source "${CONDA_HOME}/etc/profile.d/conda.sh" && \
    conda activate "${CONDA_HOME}/envs/Tensorflow" && \
    cd "${WORK_DIR}/models/research" && \
    python3 -m pip install  "${WORK_DIR}/models/research" && \
    python3 -m pip install --upgrade object-detection tensorflow tensorflow-text tensorflow-io 

    # python3 -m pip install --upgrade tensorflow tensorflow-text tensorflow-io object-detection
# Clean up
RUN apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*  /tmp/library-scripts/ /tmp/conda-tmp/ /home/tensorflow/models

# ENTRYPOINT ["python", "object_detection/model_main_tf2.py"]
# ENTRYPOINT [ "conda", "activate", "Tensorflow" ]
CMD [ "conda", "activate", "Tensorflow" ]