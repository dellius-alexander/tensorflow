# Installation and Use of Tensorflow On VSCode with Anaconda

***Note: This was conducted on linux, Ubuntu-18.04/elemtary OS 5.1.7 Hera.***

---

## 1. Download and Install Anaconda

***Note: Installation instruction can be found at [Anaconda Docs](https://docs.anaconda.com/anaconda/install/linux/)***

The below script block will handle installing and setting up Anaconda.

You can place it in file and at the terminal type `sudo bash conda.sh`

```bash
# Dependencies
apt-get install -y \
libgl1-mesa-glx \
libegl1-mesa \
libxrandr2 \
libxrandr2 \
libxss1 \
libxcursor1 \
libxcomposite1 \
libasound2 \
libxi6 \
libxtst6
wait $!
# Download Anaconda
curl -fsSLo /tmp/Anaconda3-2021.05-Linux-x86_64.sh  \
https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh
wait $!
# Locate file has here: https://docs.anaconda.com/anaconda/install/hashes/lin-3-64/
SHA256SUM="2751ab3d678ff0277ae80f9e8a74f218cfc70fe9a9cdc7bb1c137d7e47e33d53"
# Check file has after download and 
if [[ $(sha256sum Anaconda3-2021.05-Linux-x86_64.sh | awk '{print $1}') =~ ^(${SHA256SUM})$ ]]; then
    printf "\nFile verfication PASSED...\n\tActivating Conda...\n"
    source $(whereis conda | cut -d '/' -f-4 | awk '{print $2}')/bin/activate
else
    printf "\nFile has verification failed...\n"
fi

```
*To control whether or not each shell session has the base environment activated or not, run conda `config --set auto_activate_base False or True`. To run conda from anywhere without having the base environment activated by default, use conda `config --set auto_activate_base False`. This only works if you have run conda init first.*


After installation and initialization of `Conda` your terminal should be prefixed with "`(base) :-$`".<br/>
Now we can move onto installing `Tensorflow`.

---

## 2. Open VSCode and start an empty project

- VSCode Extension: to enhance the whole experience install the following extensions
    - donjayamanne.python-extension-pack
    - changkaiyan.tf2snippets (Tensorflow 2.0 Snippets)
    - ms-toolsai.jupyter
    - ms-vscode.cpptools (Microsoft C++ tools)
    - redhat.fabric8-analytics

- Add the below settings to your `~/project_folder/.vscode/settings.json` file located in your workspace. This will tell vscode to use the virtual environment interpreter and environment to run your project scripts.

    ```json
    {
    "python.defaultInterpreterPath": "${workspaceFolder}/my_venv/bin/python"
    }
    ```
- Project Directory Structure: 

    You can try any tutorial when we are complete the installation but your project should at least resemble below when finished. 

    ***Note: some tutorials will download additional dependencies.***

    ```bash
    Tensorflow   # project top folder
    |
    |___my_venv # python virtual environment
    |   |
    |   |___bin/python # python executable
    |
    |___tutorial.py    # some Tensorflow tutorial
    
    ```

- Open a terminal in your vscode project setup your virtual environment
    - Method 1: simple command line definition
    
        ```bash
        # Create python virtual environment and capture the setup logs to 
        # verify all installed packages later
        conda create --prefix=${PWD}/my_venv python=3.8 \
        anaconda tensorflow | tee ./logs/conda-venv-install.log
        ```

    - Method 2: create an `env.yaml` configuration file and define your python environment. It should look like this.
    
        ```yaml
        name: my_venv
        channels: 
        - defaults
        # Dependencies
        dependencies:
        - pip=20.0.1
        - tensorflow=2.4.1
        - anaconda=4.10.1
        # provide a directory for your new python environment
        prefix: ${PWD}/my_venv
        ```
        
        - Run the command to create the above python virtual environment from our `conda configuration file`
        
            ```bash
            # we pipe the output to enable caching of the output
            (base) :-$ conda env create -f tensorflow.yml \
            | tee ./logs/conda-venv-install.log
            ```

    - Activate/Deactivate `my_venv`:
        - To activate this environment, use:
            ```bash
            (base) :-$ conda activate ${PWD}/my_venv
            ```
        - To deactivate an active environment, use
            ```bash
            (my_venv) :-$ conda deactivate
            # conda reverts back to base
            (base) :-$
            ```
        - To deactivate conda run:
            ```bash
            (base) :-$ conda deactivate
            # terminal is no longer prepended with conda (base)
            :-$
            ```
    - To see your existing environment or verify if the new environment was created, use this command:
        ```bash
        conda info --envs
        ```

    - To delete or remove the environment, type the following in your terminal:
        ```bash
        # The --all option helps remove all packages from the environment named env_name
        conda remove --name env_name --all
        # or specify a directory of the installed virtual environment
        conda remove my_venv
        ```

    - Create an requirements file of your environment, listing all packages and dependencies. Use the file to duplicate your working environment in another project if needed.
    
        ```bash
        # conda env export > <environment name>.yml 
        conda env export > tensorflow.yml 
        ```
## 3. Verify Tensorflow installation

Now that our python environment setup, we can add TensorFlow tutorial to our python file `tutorial.py` and execute the python code. You can find a list of tutorials on the offical [TensorFlow Home](https://www.tensorflow.org/tutorials).

The below should be all you need to verify you installation of `TensorFlow`. If you receive any error their maybe some dependencies issues.

- tutorial.py
    ```python
    import tensorflow as tf
    print("TensorFlow version: ", tf.__version__)
    ```

- The above python script should return the TersonFlow version, verifying we have a stable installation.

    ```bash
    TensorFlow version: 2.4.1
    ```

Your are now ready to start using `TensorFlow`.

---

## 4. Install the Object Detection API with TensorFlow 2<a href="#" id="object-detection-api-install" ></a>

[![Python 3.6](https://img.shields.io/badge/Python-3.6-3776AB)](https://www.python.org/downloads/release/python-360/)
[![TensorFlow 2.2](https://img.shields.io/badge/TensorFlow-2.2-FF6F00?logo=tensorflow)](https://github.com/tensorflow/tensorflow/releases/tag/v2.2.0)
[![Protobuf Compiler >= 3.0](https://img.shields.io/badge/ProtoBuf%20Compiler-%3E3.0-brightgreen)](https://grpc.io/docs/protoc-installation/#install-using-a-package-manager)

## Prerequisites:
- [Tensorflow](https://www.tensorflow.org/install) must be installed in your virtual environment
- [Docker Desktop](https://www.docker.com/products/docker-desktop) must be installed on your host machine
- [Git Bash](https://git-scm.com/downloads) to clone the Models Repo needed to build the Object Detection API

## ***Automated installation***

***I have provided a script to automate the build and installation of the Object Detection API with Tensorflow 2. Simply download and execute the file within the `root` of your `project` [Click here to see install_object_detection_api_2 script.](install_object_detection_api_2.sh)***

## Manual Installation

You can install the TensorFlow Object Detection API either with Python Package
Installer (pip) or Docker. For local runs we recommend using Docker and for
Google Cloud runs we recommend using pip.

Clone the TensorFlow Models repository and proceed to one of the installation
options.

```bash
git clone https://github.com/tensorflow/models.git
```

### Docker Installation

```bash
# From the root of the git repository
docker build -f research/object_detection/dockerfiles/tf2/Dockerfile -t od .
docker run -it od
```

### Python Package Installation

```bash
cd models/research
# Compile protos.
protoc object_detection/protos/*.proto --python_out=.
# Install TensorFlow Object Detection API.
cp object_detection/packages/tf2/setup.py .
python -m pip install --use-feature=2020-resolver .
```

```bash
# Test the installation.
python object_detection/builders/model_builder_tf2_test.py
```

## Quick Start

### Colabs

<!-- mdlint off(URL_BAD_G3DOC_PATH) -->

*   Training -
    [Fine-tune a pre-trained detector in eager mode on custom data](../colab_tutorials/eager_few_shot_od_training_tf2_colab.ipynb)

*   Inference -
    [Run inference with models from the zoo](../colab_tutorials/inference_tf2_colab.ipynb)

*   Few Shot Learning for Mobile Inference -
    [Fine-tune a pre-trained detector for use with TensorFlow Lite](../colab_tutorials/eager_few_shot_od_training_tflite.ipynb)

<!-- mdlint on -->

## Training and Evaluation

To train and evaluate your models either locally or on Google Cloud see
[instructions](tf2_training_and_evaluation.md).

## Model Zoo

We provide a large collection of models that are trained on COCO 2017 in the
[Model Zoo](tf2_detection_zoo.md).

## Guides

*   <a href='configuring_jobs.md'>
      Configuring an object detection pipeline</a><br>
*   <a href='preparing_inputs.md'>Preparing inputs</a><br>
*   <a href='defining_your_own_model.md'>
      Defining your own model architecture</a><br>
*   <a href='using_your_own_dataset.md'>
      Bringing in your own dataset</a><br>
*   <a href='evaluation_protocols.md'>
      Supported object detection evaluation protocols</a><br>
*   <a href='tpu_compatibility.md'>
      TPU compatible detection pipelines</a><br>
*   <a href='tf2_training_and_evaluation.md'>
      Training and evaluation guide (CPU, GPU, or TPU)</a><br>
