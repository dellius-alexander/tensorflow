# Installation and Use of Tensorflow On VSCode with Anaconda

***Note: This was conducted on linux, Ubuntu-18.04/elemtary OS 5.1.7 Hera.***

---

## 1. Download and Install Anaconda

The below script block will handle installing and setting up Anaconda.<br/>
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
    printf "\nFile verfication PASSED...\n\tInitializing Conda...\n"
    # eval $(conda init &)
else
    printf "\nFile has verification failed...\n"
fi

```

After installation and initialization of `Conda` your terminal should be prefixed with "`(base) :-$`".<br/>
Now we can move onto installing `Tensorflow`.

---

## 2. Open VSCode and start an empty project

- VSCode Extension: to enhance the whole experience install the following extensions
    - donjayamanne.python-extension-pack
    - changkaiyan.tf2snippets
    - ms-toolsai.jupyter
    - ms-vscode.cpptools
    - redhat.fabric8-analytics

- Project Directory Structure: 

    You can try any tutorial when we are complete the installation but your project should at least resemble below when finished. 

    ***Note: some tutorials will download additional dependencies.***

    ```bash
    Tensorflow   # project top folder
    |
    |___my_venv # python virtual environment
    |   |
    |   |___bin/activate # python virtual environment activation script
    |
    |___tutorial.py    # some Tensorflow tutorial
    
    ```

- Open a terminal in your vscode project setup your virtual environment
    - Method 1: simple command line definition
        ```bash
        # Create python virtual environment and capture the setup logs to verify all installed packages later
         conda create --prefix=${PWD}/my_venv python=3.8 anaconda tensorflow | tee ./logs/conda-venv-install.log
        ```
    - Method 2: create an `env.yaml` file and define your python environment. It should look like this.
    
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
            (base) :-$ conda env create -f tensorflow.yml | tee ./logs/conda-venv-install.log
            ```

    - Activate/Deactivate `my_venv`:
        - To activate this environment, use:
            ```bash
            (base) :-$ conda activate ${PWD}/my_venv
            ```
        - To deactivate an active environment, use
            ```bash
            (base) (my_venv):-$ conda deactivate
            ```
## 3. Verify Tensorflow installation

Now that our python environment setup, we can add TensorFlow tutorial to our python file `tutorial.py` and execute the python code. You can find a list of tutorials on the offical [TensorFlow Home](https://www.tensorflow.org/tutorials)

The below should be all you need to verify you installation of `TensorFlow`. If you receive any error their maybe some dependencies issues.

- tutorial.py
    ```python
    import tensorflow as tf
    print("TensorFlow version: ", tf.__version__)
    ```

The above python script should return the TersonFlow version, verifying we have a stable installation.

```bash
TensorFlow version: 2.4.1
```

Your are now ready to start using `TensorFlow`.