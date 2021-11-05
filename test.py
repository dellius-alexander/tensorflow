# Load the TensorBoard notebook extension
# %load_ext tensorboard
# # Make numpy values easier to read.
# np.set_printoptions(precision=3, suppress=True)
import datetime
import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
import os
import fiftyone as fo
import fiftyone.zoo as foz
from tensorflow_datasets.core.as_dataframe import DataFrame
from tensorflow_datasets.core.dataset_builder import REUSE_DATASET_IF_EXISTS
import fiftyone.utils.data as foud
import numpy as np
import pandas as pd
import json
from tensorflow.keras import datasets, layers, models

print("TensorFlow version: ", tf.__version__)
# # print('{0:.2f}'.format(9.25/3))
# # import tensorflow_datasets as tfds
# #! Warning: Manually default download to ~/tensorflow_datasets/manual/imagenet2012

# # Build the dataset
# data_path = os.path.abspath("Tensorflow_datasets")
# print(data_path)
# # # Create an instance of your custom dataset importer
# # importer = foud.importers.DatasetImporter(
# #                                         data_path,
# #                                         shuffle=False,
# #                                         seed=123,
# #                                         max_samples=100)
# # # Import the dataset!
# # dataset = fo.Dataset.from_importer(importer,name="voc")

# # session = fo.launch_app(dataset)
# # session.wait()


# (dataset, dataset_info)= tfds.load(
#                                     'voc',
#                                     data_dir=data_path,
#                                     batch_size=64,
#                                     # as_supervised=True,
#                                     shuffle_files=True,
#                                     download=True,
#                                     split=["train","validation","test"],
#                                     with_info=True)
# # Normalize pixel values to be between 0 and 1
# dataset = dataset / 255.0
# print(dataset)
# # model = models.Sequential()
# # model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
# # model.add(layers.MaxPooling2D((2, 2)))
# # model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# # model.add(layers.MaxPooling2D((2, 2)))
# # model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# # # Let's display the architecture of your model so far:
# # model.summary()

# # # Add Dense layers on top
# # model.add(layers.Flatten())
# # model.add(layers.Dense(64, activation='relu'))
# # model.add(layers.Dense(10))

# # # Here's the complete architecture of your model:
# # model.summary()

# # Compile and train the modelprefix: ${PWD}/my_venv
# # model.compile(optimizer=)

# # ds_numpy = tfds.as_numpy(dataset)
# # for ex in ds_numpy:
# #     dataset_json = json.dumps(str(ex),indent=4,separators=(',',':'),allow_nan=True )
# # print(str(dataset_json))


# # # for ex in ds_numpy:
# # #     print(ex)
# # #     for e in ex:
# # #         print(e)



# # # df = tfds.as_dataframe()
# # # Normalize pixel values to be between 0 and 1
# # # train_images, test_images = train_images / 255.0, test_images / 255.0


# # # df = tfds.as_dataframe(ds=dataset, ds_info=dataset_info)

# # #
# # # dataset = tfds.as_dataframe(
# #     dataset,
# #     dataset_info
# #     )
# #
# # tfds.visualization.show_examples(
# #     ds=df,
# #     ds_info=dataset_info
# #     )

# # session = fo.launch_app(dataset)
# # session.wait()
