from json.decoder import JSONDecoder
from json.encoder import JSONEncoder
import os, tensorflow_datasets as tfds,\
    platform,socket,re,uuid,json,psutil,logging,\
        pandas as pd, re, lxml
from object_detection.dataset_tools import create_pascal_tf_record as cptr
from typing import List
# You have to build the object detection API before you can use 
# model builder.
import object_detection as od
from object_detection.builders import dataset_builder as dbr
from object_detection.builders import model_builder as mb
from object_detection.utils import config_util as cu, dataset_util, label_map_util
from object_detection.protos import model_pb2 as mpb2
from tensorflow_datasets.core import utils
from tensorflow_datasets.core.proto.dataset_info_generated_pb2 import DatasetInfo
from lxml import etree
from object_detection.protos.string_int_label_map_pb2 import StringIntLabelMap, StringIntLabelMapItem
from google.protobuf import text_format
import tensorflow_datasets as tfds
import tensorflow as tf
#####################################################################
# Setup Logging
logging.basicConfig(format='%(asctime)s %(message)s',filename="tensorflow.log", datefmt='%m/%d/%Y %I:%M:%S %p',level=logging.DEBUG)
logging.debug('This message is a test for log level...DEBUG')
logging.info('This message is a test for log level...INFO')
logging.warning('This message is a test for log level...WARNING')
logging.error('This message is a test for log level...ERROR')
#####################################################################
def convert_classes(classes, start=1):
    """
    Converting python list to label_map.pbtxt
    """
    msg = StringIntLabelMap()
    for id, name in enumerate(classes, start=start):
        msg.item.append(StringIntLabelMapItem(id=id, name=name))

    text = str(text_format.MessageToBytes(msg, as_utf8=True), 'utf-8')
    return text

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
####################################################################
def get_dataset_info_as_json(info=DatasetInfo):
    info_json = {
        'name': info.name,
        'full_name': info.full_name,
        'description': info.description,
        'homepage': info.homepage,
        'data_path': info.data_dir,
        'download_size': info.download_size,
        'dataset_size': info.dataset_size,
        'features': info.features.to_json(),
        'supervised_keys': info.supervised_keys,
        'disable_shuffling': info.disable_shuffling,
        'splits': json.dumps(str(info.splits.items()),check_circular=True,indent=4,separators=(',',':'),cls=JSONEncoder),
        'citation': info.citation
    }
    info_json = json.dumps(info_json,check_circular=True,indent=4,separators=(',',':'))
    # print(info_json)
    logging.info(info_json)
    open("DatasetInfo.json","wb").write(bytes(info_json,"utf-8"))
    return info_json
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
# get the newly creaed dataset directory path
data_path = info.data_dir
# create json file of the dataset info and [optionally print info]
# logging.info(get_dataset_info_as_json(info))
# train_dir = "Tensorflow_datasets/downloads/extracted/VOC2007_train"
# test_dir = "Tensorflow_datasets/downloads/extracted/VOC2007_test"
# train_annot_dir = train_dir+"/VOCdevkit/VOC2007/Annotations"
# train_image_dir = train_dir+"/VOCdevkit/VOC2007/JPEGImages"
train_data_path = "Tensorflow_datasets/downloads/extracted/VOC2007_train/VOCdevkit"
test_data_path = "Tensorflow_datasets/downloads/extracted/VOC2007_test/VOCdevkit"
output_path = "Tensorflow_datasets/data"
label_directory = "Tensorflow_datasets/data/label_map.pbtxt"
raw_labels = "Tensorflow_datasets/voc/2007/4.0.0/labels.labels.txt"

#####################################################################
# serialize the classes from the raw labels file and convert to bytes
## and write the serialized list of classes to file
#####################################################################
### get the contents of the labels file
my_file = open(raw_labels, "r")
### read the contents of labels file
content = my_file.read()
### convert the contents to list
class_label_list = content.split("\n")
my_file.close() # Close the file
logging.info("""
Class List:\n{0}
""".format(class_label_list))
### serialize the list of classes
class_bytes_label_serialized = convert_classes(class_label_list)
# write the converted classes to label_map.pbtxt file
with open(label_directory, 'w') as f:
        f.write(class_bytes_label_serialized)
#####################################################################
#####################################################################
##### Convert raw PASCAL dataset to TFRecord for object_detection.
####### Models based on the TensorFlow object detection API need a 
####### special format for all input data, called "TFRecord".
#####################################################################
# A TFRecord file stores your data as a sequence of binary strings. 
# This means you need to specify the structure of your data before 
# you write it to the file. Tensorflow provides two components for 
# this purpose: tf.train.Example and tf.train.SequenceExample. You 
# have to store each sample of your data in one of these structures, 
# then serialize it and use a tf.python_io.TFRecordWriter to write it 
# to disk.
#####################################################################
# (module) create_pascal_tf_record
# Convert raw PASCAL dataset to TFRecord for object_detection.
# Example usage:
#     python object_detection/dataset_tools/create_pascal_tf_record.py
#         --label_map_path=/home/user/VOCdevkit/label_map*
#         --data_dir=/home/user/VOCdevkit
#         --year=VOC2012
#         --set=["train","test"]
#         --output_path=/home/user/pascal.record
#         --category=["cat","dog"]
#
# capture return variable; it should be 0 if all went well...
rtn = os.system('''python3 Models/research/object_detection/dataset_tools/create_pascal_tf_record.py \
        --data_dir={0} \
        --year={1} \
        --set={2} \
        --output_path={3} \
        --category={4} \
        --label_map_path={5}'''.
    format(
        train_data_path,  #   --data_dir=/home/user/VOCdevkit
        "VOC2007",      #   --year=VOC2012
        "train",        #   --set=["train","test"]
        (output_path+"/train_pascal.record"),    #   --output_path=/home/user/pascal.record
        class_label_list,    #   --category=["cat","dog"]
        label_directory #   --label_map_path=/home/user/voc_opath/label_map.pbtxt*
    ))
print(rtn)
rtn = os.system('''python3 Models/research/object_detection/dataset_tools/create_pascal_tf_record.py \
        --data_dir={0} \
        --year={1} \
        --set={2} \
        --output_path={3} \
        --category={4} \
        --label_map_path={5}'''.
    format(
        test_data_path,  #   --data_dir=/home/user/VOCdevkit
        "VOC2007",      #   --year=VOC2012
        "test",        #   --set=["train","test"]
        (output_path+"/test_pascal.record"),    #   --output_path=/home/user/pascal.record
        class_label_list,    #   --category=["cat","dog"]
        label_directory #   --label_map_path=/home/user/voc_opath/label_map.pbtxt*
    ))
print(rtn)
with open('Tensorflow_datasets/voc/2007/4.0.0/features.json') as file:
    features = json.load(file,cls=JSONDecoder)
    features_json = json.dumps(features,check_circular=True,indent=4)
print(features_json)
# detection_model = mb.build(cu.model_pb2.DetectionModel(),is_training=True)

# print(detection_model)
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