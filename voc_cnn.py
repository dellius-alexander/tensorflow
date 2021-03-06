#@author: Dellius Alexander
#@Professor: Ken
#@Date: 11/05/2021
#@Description: The below is a custom modeling and visualization of CNN Object Detection Model.
##########################################################################
from json import encoder
from json.decoder import JSONDecoder
from json.encoder import JSONEncoder
import os, tensorflow_datasets as tfds,\
    platform,socket,re,uuid,json,psutil,logging,\
        pandas as pd, re, lxml
# from object_detection.dataset_tools import create_pascal_tf_record as cptr
# from typing import List
# You have to build the object detection API before you can use 
# model builder.
# import object_detection as od
# from object_detection.builders import dataset_builder as dbr
# from object_detection.builders import model_builder as mb
# from object_detection.protos import model_pb2 as mpb2
# from tensorflow_datasets.core import utils
from tensorflow_datasets.core.proto.dataset_info_generated_pb2 import DatasetInfo
# from lxml import etree
from object_detection.protos.string_int_label_map_pb2 import StringIntLabelMap, StringIntLabelMapItem
from google.protobuf import text_format
import tensorflow_datasets as tfds
# import tensorflow as tf
import fiftyone as fo
# from fiftyone.utils import data, tf as fotf
#####################################################################
#####################################################################
#####################################################################
# Setup Logging
# logging.basicConfig(filename="./tensorflow.log",format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',level=logging.DEBUG)
# create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# create console handler and file handler; and set level to debug
# ch = logging.StreamHandler()
fh = logging.FileHandler("./tensorflow.log",mode='w')
fh .setLevel(logging.DEBUG)
# ch.setLevel(logging.DEBUG)
# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# add formatter to ch and fh
fh.setFormatter(formatter)
# ch.setFormatter(formatter)

# add ch and fh to logger
# logger.addHandler(ch)
logger.addHandler(fh)
logger.debug('This message is a test for log level...DEBUG')
logger.info('This message is a test for log level...INFO')
logger.warning('This message is a test for log level...WARNING')
logger.error('This message is a test for log level...ERROR')
#####################################################################
#####################################################################
def convert_classes_to_bytes(classes, start=1, encoding='utf-8'):
    """
    Converting python list to label_map.pbtxt
    ;param classes; your list of classes or category
    ;param start; start index
    :return list[str(text_format.MessageToBytes(msg, as_utf8=True), 'utf-8')]
    """
    msg = StringIntLabelMap()
    for id, name in enumerate(classes, start=start):
        msg.item.append(StringIntLabelMapItem(id=id, name=name))
    text = str(text_format.MessageToBytes(msg, as_utf8=True), encoding)
    return text
#####################################################################
#####################################################################
#Shamelessly combined from google and other stackoverflow like 
# sites to form a single function
def getSystemInfo():
    '''Gets the current system|machine information
    :return dict[]'''
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
        logger.exception(e)
#####################################################################
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
        logging.info("----------------- Created Dataset Directory: ")
        data_path = os.path.abspath(data_path)
        logging.info(data_path)
        logging.info("--------------------------------------------")
        return data_path
    else:
        data_path = os.path.abspath(data_path)
        logging.info("------------------- Found Dataset Directory: ")
        logging.info(data_path)
        logging.info("--------------------------------------------")
        return data_path
#####################################################################
#####################################################################
def get_dataset(dataset_name,dataset_directory,batch_size):
    """
    Checks for and creates the dataset directory if it does
    not exist.
    :param dataset_name: the dataset name, i.e. 'voc'
    :param dataset_directory: the dataset directory
    :param batch_size: the number of training examples utilized in one iteration.
    :return: tuple[dict | list | tuple | Any, DatasetInfo] | dict | list | tuple | Any
    """
    # Create the dataset directory and download dataset
    data_path = get_dataset_dir(dataset_directory)
    ## Create dataset directory if not exist
    if not os.path.isdir(data_path):
        os.mkdir(data_path)
        logger.info("-------------- Created Dataset Directory: ")
        data_path = os.path.abspath(data_path)
        # logging.info(data_path)
        return tfds.load(
                        dataset_name,
                        data_dir=data_path,
                        batch_size=batch_size,
                        shuffle_files=True,
                        download=True,
                        split=["train","validation","test"],
                        with_info=True
                        )
    else:
        logger.info("-------------- Dataset Directory Exists: ")
        # logging.info(data_path)
        return tfds.load(
                        dataset_name,
                        data_dir=data_path,
                        batch_size=batch_size,
                        shuffle_files=True,
                        download=True,
                        split=["train","validation","test"],
                        with_info=True
                        )
#####################################################################
#####################################################################
def get_dataset_info_as_json(info=DatasetInfo) -> str:
    '''
    :param info: dataset info
    :return json:
    '''
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
        'splits': json.dumps(str(info.splits.items()),check_circular=True,\
            indent=4,separators=(',',':'),cls=JSONEncoder),
        'citation': info.citation
    }
    info_json = json.dumps(info_json,check_circular=True,indent=4,separators=(',',':'))
    # logging.info(info_json)
    logger.info("Writing info json to file: DatasetInfo.json")
    logger.info(info_json)
    open("DatasetInfo.json","wb").write(bytes(info_json,"utf-8"))
    return info_json
#####################################################################
#####################################################################
def get_dataset_features(path_to_features_json_file):
    '''
    :param path_to_features_json_file: path to features json file.
    :return tuple[dict,json]:
    '''
    with open(path_to_features_json_file) as file:
        features = json.load(file,cls=JSONDecoder)
        features_json = json.dumps(features,check_circular=True,indent=4)
    return (features,features_json)
#####################################################################
#####################################################################
def my_create_pascal_dataset_records(big_data=dict):
    '''(module) create_pascal_tf_record.
    Convert raw PASCAL dataset to TFRecord for object_detection.
    Models based on the TensorFlow object detection API need a 
    special format for all input data, called "TFRecord".
    A TFRecord file stores your data as a sequence of binary strings. 
    This means you need to specify the structure of your data before 
    you write it to the file. Tensorflow provides two components for 
    this purpose: tf.train.Example and tf.train.SequenceExample. You 
    have to store each sample of your data in one of these structures, 
    then serialize it and use a tf.python_io.TFRecordWriter to write it 
    to disk.
    Example usage:
        python object_detection/dataset_tools/create_pascal_tf_record.py
            --label_map_path=/home/user/VOCdevkit/label_map*
            --data_dir=/home/user/VOCdevkit
            --year="VOC2012"
            --set=["train","test"]
            --output_path=/home/user/pascal.record
            --category=["cat","dog"]
    -----------------------------------------------------------------
    :param big_data: dict object holding all dataset info
    '''
    #####################################################################
    #####################################################################
    # capture return variable; it should be 0 if all went well...
    rtn = os.system('''python {6} \
            --data_dir={0} \
            --year={1} \
            --set={2} \
            --output_path={3} \
            --category={4} \
            --label_map_path={5}'''.
        format(
            #   --data_dir=/home/user/VOCdevkit
            big_data['train data dir'],
            #   --year=VOC2012
            "VOC2007",
            #   --set=["train","test"]
            "train",
            #   --output_path=/home/user/pascal.record
            (big_data['output data path']+"/train_pascal.record"),
            #   --category=["cat","dog"]
            big_data['class label list'],
            #   --label_map_path=/home/user/voc_opath/label_map.pbtxt*
            big_data['label map path'],
            #   create tensorflow record
            big_data['create_pascal_tf_record']
        ))
    if rtn != 0:
        exit(0)
    else:
        logger.info("\nSuccess building training model...\nReturn code: %s\n",rtn)
    rtn = os.system('''python3 {6} \
            --data_dir={0} \
            --year={1} \
            --set={2} \
            --output_path={3} \
            --category={4} \
            --label_map_path={5}'''.
        format(
            #   --data_dir=/home/user/VOCdevkit
            big_data['test data dir'],
            #   --year=VOC2012
            "VOC2007",
            #   --set=["train","test"]
            "test",
            #   --output_path=/home/user/pascal.record
            (big_data['output data path']+"/test_pascal.record"),
            #   --category=["cat","dog"]
            big_data['class label list'],
            #   --label_map_path=/home/user/voc_opath/label_map.pbtxt*
            big_data['label map path'],
            #   create tensorflow record
            big_data['create_pascal_tf_record']
        ))
    if rtn != 0:
        exit(0)
    else:
        logger.info("\nSuccess building testing model...\nReturn code: %s\n",rtn)
#####################################################################
#####################################################################
def get_labels_contents_from_file(labels_raw_path,label_map_path):
    '''serialize the classes from the raw labels file and convert to 
    bytes and write the serialized list of classes to file.
    Get the list of labels from raw file, serialize and 
    return serialized list.
    :param labels_raw_path: path to raw labels input file
    :param label_map_path: path to map file output
    :return (labels_serialized_to_bytes, class_label_list):
    '''
    ### get the contents of the labels file
    my_file = open(labels_raw_path, "r")
    ### read the contents of labels file
    content = my_file.read()
    ### convert the contents to list
    class_label_list = content.split("\n")
    my_file.close() # Close the file
    logger.info("""
    Class List:\n%s
    """,(class_label_list))
    ### serialize the list of classes
    labels_serialized_to_bytes = convert_classes_to_bytes(class_label_list)
    # write the converted classes to label_map.pbtxt file
    with open(label_map_path, 'w') as f:
        f.write(labels_serialized_to_bytes)
    return (labels_serialized_to_bytes, class_label_list)
#####################################################################
#####################################################################
def check_dir(directory):
    '''Checks if the directory exist; creates the directory if it \
        if it does not exist.
    :param directory: [os.path] - the directory to be verified
    '''
    cnt = 0
    contents = {}
    if os.path.exists(directory):
        contents[directory] = {}
        for root, dirs, files in os.walk(
            top=os.path.relpath(directory),
            topdown=False
            ):
            dir_path = ""
            if len(dirs) > 1:
                cnt+=1
                for _dir in dirs:
                    dir_path = os.path.join(root,_dir)
                    print(dir_path)
                    contents[directory][_dir] = []
                    for file in files:
                        contents[directory][_dir].append(file)
                    check_dir(dir_path)
            elif len(dirs) == 1:
                print (root)
                print (dirs)
                print (files)
                print ('--------------------------------')
                cnt+=1
            if cnt == 3:
                print(contents)
                return contents

            # if len(dirs) > 1:
            #     for dir_ in dirs:
            #         contents[str(directory)][str(dir_)] = {}
                # logger.info(contents)
                    # return check_dir(os.path.join(root,dir_))
            # if len(dirs) == 1:
            #     contents[directory][dirs[0]] = []
            #     for dir in dirs:
            #         if len(files) > 0:
            #             for file in files:
            #                 contents[directory][dir].append(file)
            #             logger.info(contents[directory][dir])
            #             return contents[directory][dir]
        print(contents)
        return contents
    elif not os.path.exists(directory):
        os.mkdir(directory)
        return os.path.abspath(directory)
#####################################################################
#####################################################################
def main():
    BIG_DATA={}
    check_dir('Tensorflow_datasets')
    exit(0)
    #####################################################################
    #####################################################################
    # get the system info
    BIG_DATA['system info'] = getSystemInfo()
    #####################################################################
    # Get the dataset
    ## Define regex for the platform OS
    plat = re.compile('({0})'.format(BIG_DATA['system info']['platform']),re.IGNORECASE)
    logging.info("--------------------------------------------")
    # path separator
    ps =""
    if plat.match("Linux"):
        BIG_DATA['path separator']="/"
        print('''\n\'Platform\': \'{0}\'\n'''.format(BIG_DATA['system info']['platform']))
    if plat.match("Window"):
        BIG_DATA['path separator']="\\"
        print('''\n\'Platform\': \'{0}\'\n'''.format(BIG_DATA['system info']['platform']))
    #####################################################################
    # name of the dataset
    BIG_DATA['dataset name'] = "voc"
    # name a dataset directory
    BIG_DATA['dataset dir name'] = os.path.abspath("Tensorflow_datasets")
    
    # create dataset of batch size 32/64/128/
    BIG_DATA['dataset'], BIG_DATA['dataset info'] = get_dataset(BIG_DATA['dataset name'],BIG_DATA['dataset dir name'],32)
    # define the rest of the objects
    BIG_DATA['train data dir'] = "Tensorflow_datasets/downloads/extracted/VOC2007_train/VOCdevkit"
    BIG_DATA['test data dir'] = "Tensorflow_datasets/downloads/extracted/VOC2007_test/VOCdevkit"
    BIG_DATA['features path'] = "Tensorflow_datasets/voc/2007/4.0.0/features.json"
    BIG_DATA['output data path'] = "Tensorflow_datasets/data"
    BIG_DATA['label map path'] = "Tensorflow_datasets/data/label_map.pbtxt"
    BIG_DATA['label raw path'] = "Tensorflow_datasets/voc/2007/4.0.0/objects-label.labels.txt"
    BIG_DATA['features'] = get_dataset_features(BIG_DATA['features path'])
    BIG_DATA['seralized labels'],BIG_DATA['class label list'] = get_labels_contents_from_file(
        BIG_DATA['label raw path'],BIG_DATA['label map path'])
    BIG_DATA['create_pascal_tf_record'] = os.path.abspath("create_pascal_tf_record.py")
    BIG_DATA['tf_record_creation_util'] = os.path.abspath("tf_record_creation_util.py")
    BIG_DATA['fiftyone dir'] = os.path.abspath("fiftyone")
    
    # get the newly creaed dataset directory path
    BIG_DATA['dataset dir'] = BIG_DATA['dataset info'].data_dir
    # create json file of the dataset info and [optionally print info]
    logger.info(get_dataset_info_as_json(BIG_DATA['dataset info']))
    #####################################################################
    # logging.info(BIG_DATA['dataset info'])
    logging.info(BIG_DATA['class label list'])
    # now create the Tensorflow Record from out raw data
    my_create_pascal_dataset_records(BIG_DATA)
    # then get the Tensorflow Records data and convert to 
    # fiftyone.Dataset of type Tensorflow Object Detection Dataset
    ### https://voxel51.com/docs/fiftyone/user_guide/using_datasets.html#datasets
    #####################################################################
    # Adds the contents of the given archive to the dataset.
    # If a directory with the same root name as archive_path exists, 
    # it is assumed that this directory contains the extracted contents 
    # of the archive, and thus the archive is not re-extracted. 
    # See this guide <loading-datasets-from-disk> for example usages 
    # of this method and descriptions of the available dataset types.
    # 
    # Args:
    dataset1 = fo.Dataset("voc2007").add_archive(
        # archive_path: the path to an archive of a dataset directory
        archive_path=BIG_DATA['output data path'],
        # dataset_type (None): the fiftyone.types.dataset_types.Dataset type of the 
        # dataset in archive_path. Since we converted it using the object detection 
        # API to a TFRecord, we need to use the dataType "TFObjectDetectionDataset"
        # See:  https://voxel51.com/docs/fiftyone/api/fiftyone.types.dataset_types.html?\highlight=dataset_type#module-fiftyone.types.dataset_types
        dataset_type=fo.types.dataset_types.TFObjectDetectionDataset,
        # dataset_type=fo.types.dataset_types.TFImageClassificationDataset,
        # data_path (None): an optional parameter that enables explicit
        #   control over the location of the exported media for certain dataset 
        #   types. Can be any of the following:
        #       - a folder name like `"data"` or `"data/"` specifying a
        #       subfolder of dataset_dir in which the media lies - an absolute 
        #       directory path in which the media lies. In this case, the archive_path 
        #       has no effect on the location of the data - a filename like "data.json" 
        #       specifying the filename of a JSON manifest file in archive_path that maps 
        #       UUIDs to media filepaths. Files of this format are generated when passing 
        #       the export_media="manifest" option to fiftyone.core.collections.SampleCollection.export 
        #       - an absolute filepath to a JSON manifest file. In this case, archive_path has no 
        #       effect on the location of the data
        #       By default, it is assumed that the data can be located in the default 
        #       location within archive_path for the dataset type
        # data_path=BIG_DATA['fiftyone dir'],
        # labels_path (None): an optional parameter that enables explicit
        #       control over the location of the labels. Only applicable when importing 
        #       certain labeled dataset formats. Can be any of the following:
        #       - a type-specific folder name like `"labels"` or
        #       "labels/" or a filename like "labels.json" or "labels.xml" specifying 
        #       the location in archive_path of the labels file(s) - an absolute directory 
        #       or filepath containing the labels file(s). In this case, archive_path has 
        #       no effect on the location of the labels
        #       For labeled datasets, this parameter defaults to the location in archive_path 
        #       of the labels for the default layout of the dataset type being imported
        # labels_path=BIG_DATA['output data path'],
        # label_field (None): controls the field(s) in which imported labels
        #       are stored. Only applicable if dataset_importer is a 
        #       fiftyone.utils.data.importers.LabeledImageDatasetImporter or 
        #       fiftyone.utils.data.importers.LabeledVideoDatasetImporter. If the importer 
        #       produces a single fiftyone.core.labels.Label instance per sample/frame, this 
        #       argument specifies the name of the field to use; the default is "ground_truth".
        #       If the importer produces a dictionary of labels per sample, this argument specifies 
        #       a string prefix to prepend to each label key; the default in this case is to directly 
        #       use the keys of the imported label dictionaries as field names
        # tags (None): an optional tag or iterable of tags to attach to each
        #       sample
        # expand_schema (True): whether to dynamically add new sample fields
        #       encountered to the dataset schema. If False, an error is raised if a 
        #       sample's schema is not a subset of the dataset schema
        # add_info (True): whether to add dataset info from the importer (if
        #       any) to the dataset's info
        # image_dir (None): this parameter customize the path to ~/Tensorflow/fiftyone/
        # cleanup (True): whether to delete the archive after extracting it
        # **kwargs: optional keyword arguments to pass to the constructor of
        # the fiftyone.utils.data.importers.DatasetImporter for the specified dataset_type

    )
    # verify the dataset bytes
    # logger.info(dataset1)
    # verify the dataset has loaded
    logger.info("Dataset: %s",fo.list_datasets())
    # next load the dataset
    dataset2 = fo.load_dataset("voc2007")
    logger.info("\nCompleted Loading Dataset...\n")
    # now visualize the data and see the class mapping 
    # with boxes
    session = fo.launch_app(dataset2,address='127.0.0.1',port=5151)
    session.wait()
#####################################################################
#####################################################################
#*******************************************************************#
############################    MAIN    #############################
#####################################################################
if __name__ == '__main__':
    main()