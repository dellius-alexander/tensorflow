from json.decoder import JSONDecoder
from json.encoder import JSONEncoder
import os, tensorflow_datasets as tfds,\
    platform,socket,re,uuid,json,psutil,logging,\
        pandas as pd, re
from typing import List

#####################################################################
#Shamelessly combined from google and other stackoverflow like 
# sites to form a single function
def getSystemInfo():
    try:
        info={}
        info['platform']=platform.system()
        info['platform-release']=platform.release()
        info['platform-version']=platform.version()
        info['architecture']=platform.machine()
        info['hostname']=socket.gethostname()
        info['ip-address']=socket.gethostbyname(socket.gethostname())
        info['mac-address']=':'.join(re.findall('..', '%012x' % uuid.getnode()))
        info['processor']=platform.processor()
        info['ram']=str(round(psutil.virtual_memory().total / (1024.0 **3)))+" GB"
        return info
        # return (json.loads(json.dumps(info)))
    except Exception as e:
        logging.exception(e)
#####################################################################
def get_dataset_dir(dataset_directory) -> os.path:
    """
    Creates the data directory if it does not exist. 
    :param dataset_directory:
    :return: the absolute path of the dataset directory
    """
    # Create the dataset directory and download dataset
    data_path = dataset_directory
    if not os.path.isdir(data_path):
        os.mkdir(data_path)
        print("----------------- Created Dataset Directory: ")
        data_path = os.path.abspath(data_path)
        print(data_path)
        print("--------------------------------------------")
        return data_path
    else:
        data_path = os.path.abspath(data_path)
        print("------------------- Found Dataset Directory: ")
        print(data_path)
        print("--------------------------------------------")
        return data_path
#####################################################################
def get_dataset(dataset_directory,batch_size):
    """
    Checks for and creates the dataset directory if it does
    not exist.
    :param dataset_directory: the dataset directory
    :return: tuple[dict | list | tuple | Any, DatasetInfo] | dict | list | tuple | Any
    """
    # Create the dataset directory and download dataset
    data_path = get_dataset_dir(dataset_directory)
    ## Create dataset directory if not exist
    if not os.path.isdir(data_path):
        os.mkdir(data_path)
        print("-------------- Created Dataset Directory: ")
        data_path = os.path.abspath(data_path)
        # print(data_path)
        return tfds.load(
                        'voc',
                        data_dir=data_path,
                        batch_size=batch_size,
                        shuffle_files=True,
                        download=True,
                        split=["train","validation","test"],
                        with_info=True,
                        )
    else:
        # print(data_path)
        return tfds.load(
                        'voc',
                        data_dir=data_path,
                        batch_size=batch_size,
                        shuffle_files=True,
                        download=True,
                        split=["train","validation","test"],
                        with_info=True
                        )
#####################################################################
# get the system info
df = getSystemInfo()
#####################################################################
# Get the dataset
## Define regex for the platform OS
plat = re.compile('({0})'.format(df['platform']),re.IGNORECASE)
print("--------------------------------------------")
# path separator
ps =""
if plat.match("Linux"):
    ps="/"
    print('\'Platform\': \'{0}\''.format(df['platform']))
if plat.match("Window"):
    ps="\\"
    print('\'Platform\': \'{0}\''.format(df['platform']))
#####################################################################
# name a dataset directory
data_dir_name = "Tensorflow_datasets"
# create dataset of batch size 32/64/128/
dataset, info = get_dataset(data_dir_name,32)
data_path = get_dataset_dir(data_dir_name)
print( info.as_json)
open("DatasetInfo.json","wb").write(info.as_json.encode())
#####################################################################
# Tensorflow dataset tools
# https://github.com/tensorflow/models/tree/master/research/object_detection/dataset_tools
rtn = os.system('python3 create_pascal_tf_record.py \
        --data_dir={0}{1} \
        --year={2} \
        --output_path={0}{1}'.
        format(
            info.data_dir,  # "/home/user/VOCdevkit",
            ps,             # path separator
            "VOC2007"       # "VOC2012",
                            # "/home/user/pascal.record"
        ))
print(rtn)


# # Normalize pixel values to be between 0 and 1
# # dataset = dataset / 255.0
# for d in dataset:
#     print([str(tfds.as_numpy(a)) for a in (d)])



# model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# # Let's display the architecture of your model so far:
# model.summary()

# # Add Dense layers on top
# model.add(layers.Flatten())
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(10))

# # Here's the complete architecture of your model:
# model.summary()

# session = fo.launch_app(dataset)
# session.wait()