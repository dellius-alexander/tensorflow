#!/usr/bin/env bash
#                       PARAMS:
# Syntax: ./anaconda.sh \
#          Parameters:  [[0][-b {LICENSE MOFE flag}]], 
#                       [[1][-p {PREFIX flag}]], 
#                       [[2][PATH to ${CONDA_HOME}]], 
#                       [[3][PATH to your app directory]], 
#                       [[4][USERNAME]],
#                       [[5][ENV_FILE]]
set -e
###########################################################
RED='\033[0;31m' # Red
NC='\033[0m' # No Color CAP
# ###########################################################
LICENSE_MODE=${1:-"-b"} # Assumes you agree to license when enabled, and will not prompt stdin
PREFIX=${2:-"-p"}
CONDA_HOME=${3:-"/home/tensorflow/anaconda3"}
APP_DIR=${4-"/home/tensorflow/Tensorflow"}
USERNAME=${5:-"dalexander"}
ENV_FILE=${6:-"/tmp/conda-tmp/environment.yml"}
#
printf "${RED}\n{\n\
\t\"LICENSE_MODE\": \"${LICENSE_MODE}\",\n\
\t\"PREFIX\": \"${PREFIX}\",\n\
\t\"CONDA_HOME\": \"${CONDA_HOME}\",\n\
\t\"APP_DIR\": \"${APP_DIR}\",\n\
\t\"USERNAME\": \"${USERNAME}\",\n\
\t\"ENV_FILE\": \"${ENV_FILE}\"\n\
\n}\n${NC}"
#
wait $!
printf "${RED}\n\nInstalling Anaconda dependencies......\n\n${NC}"
sleep 1
# Dependencies
apt-get update -y && apt-get install -y \
libgl1-mesa-glx \
libegl1-mesa \
libxrandr2 \
libxrandr2 \
libxss1 \
libxcursor1 \
libxcomposite1 \
libasound2 \
libxi6 \
libxtst6 \
python3-pip
wait $!
# Download Anaconda
curl -fsSLo /tmp/Anaconda3-2021.05-Linux-x86_64.sh  \
https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh
wait $!
# Locate file has here: https://docs.anaconda.com/anaconda/install/hashes/lin-3-64/
SHA256SUM="2751ab3d678ff0277ae80f9e8a74f218cfc70fe9a9cdc7bb1c137d7e47e33d53"
# # Check file has after download and 
if [[ $(sha256sum /tmp/Anaconda3-2021.05-Linux-x86_64.sh | awk '{print $1}') =~ ^(${SHA256SUM})$ ]]; then
    printf "\n${RED}File verfication PASSED...\n\tActivating Conda...${NC}\n"
    # To run the silent installation of Miniconda for macOS or Linux, specify the -b and -p arguments of the bash installer. The following arguments are supported:
    # -b — Batch mode with no PATH modifications to ~/.bashrc. Assumes that you agree to the license agreement. Does not edit the .bashrc or .bash_profile files.
    # -p — Installation prefix/path.
    # -f — Force installation even if prefix -p already exists.
    bash /tmp/Anaconda3-2021.05-Linux-x86_64.sh "-b" "-p" "${CONDA_HOME}" "-f" &&
    wait $!  &&
    printf "\n${RED}Anaconda has been install at: \n\t[ CONDA_HOME=\"${CONDA_HOME}\" ]${NC}\n\n" &&
    export PYTHON_INTERPRETER_PATH="${CONDA_HOME}/envs/Tensorflow/bin/python3.8" &&
    export TENSORFLOW_CONDA_ENV="${CONDA_HOME}/envs/Tensorflow/" &&
    # setup user permissions for anaconda directory
    # chown ${USERNAME}:${USERNAME} -R ${CONDA_HOME} &&
    # REMEMBER: Install conda in user home, i.e: /home/john/anaconda3
    source "${CONDA_HOME}/etc/profile.d/conda.sh" && 
    for file in "~/.bashrc" "~/.zshrc"; 
    do 
    if [[ -f "${file}"  ]]; then
    cat >>$file<<EOF
    #####################################################################
    # Python Alias
    alias python="${CONDA_HOME}/envs/Tensorflow/bin/python3.8"
    #chsh -s /bin/bash

    # >>> conda initialize >>>
    # !! Contents within this block are managed by 'conda init' !!
    __conda_setup='$("${CONDA_HOME}/bin/conda" "shell.${SHELL}" "hook" 2> /dev/null)'
    if [ $? -eq 0 ]; then
    eval "$__conda_setup"
    else
    if [ -f "${CONDA_HOME}/etc/profile.d/conda.sh" ]; then
    . "${CONDA_HOME}/etc/profile.d/conda.sh"  # commented out by conda initialize
    else
        export PATH="${CONDA_HOME}/bin:${PATH}"
    fi
    fi
    unset __conda_setup
    #<<< conda initialize <<<
    ####################################################################
EOF
fi;
done;
    # usage: conda update [-h] [-n ENVIRONMENT | -p PATH] [-c CHANNEL] [--use-local]
    #                 [--override-channels] [--repodata-fn REPODATA_FNS]
    #                 [--strict-channel-priority] [--no-channel-priority]
    #                 [--no-deps | --only-deps] [--no-pin] [--copy] [-C] [-k]
    #                 [--offline] [-d] [--json] [-q] [-v] [-y] [--download-only]
    #                 [--show-channel-urls] [--file FILE] [--force-reinstall]
    #                 [--freeze-installed | --update-deps | -S | --update-all | --update-specs]
    #                 [--clobber]
    #                 [package_spec [package_spec ...]]
    # conda update -n base -c defaults conda 
    wait $!
    # [[ $? != 0 ]] && echo $? && exit 0

else
    printf "${RED}\nFile has verification failed...\n${NC}"
    exit 1
fi
# create our environment from  file and capture the logs for later
# ENV_FILE="/tmp/library-scripts/env.tensorflow.yaml"
if [[ -f "${ENV_FILE}" ]]; then
# if [[ -f "./env.tensorflow.yaml" ]]; then
    # source "${CONDA_HOME}/etc/profile.d/conda.sh" && 
    printf "${RED}\nInstalling Tensorflow at: \n{\n \t\"env_name\": \"Tensorflow\", \n\t\"path\": \"${TENSORFLOW_CONDA_ENV}\" \n}\n\n${NC}"
        # Start a new shell
    $shell
    ln -s ${CONDA_HOME}/condabin/conda /usr/local/bin &&
    conda update -n base -c defaults conda && 
    conda env create -f "${ENV_FILE}"  --prefix=${TENSORFLOW_CONDA_ENV} &&
    # conda env create -f "${ENV_FILE}" --prefix="${TENSORFLOW_CONDA_ENV}" &&
    # $shell && 
    # activate Tensorflow environment
    conda activate "${CONDA_HOME}/envs/Tensorflow" && 
    export PYTHONPATH="${CONDA_HOME}/envs/Tensorflow/bin/python3.8" && 
    python3 -m pip install tensorflow_datasets fiftyone && 
    # python3 -m pip install --use-feature=2020-resolver /home/tensorflow/models/research
    printf "${RED}\nTensorflow has been install at: \n{\n \t\"env_name\": \"Tensorflow\", \n\t\"path\": \"${TENSORFLOW_CONDA_ENV}\" \n}\n\n${NC}"
    # [[ $? != 0 ]] && echo $? && exit 1
fi