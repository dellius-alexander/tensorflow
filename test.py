import tensorflow as tf, os

print("TensorFlow version: ", tf.__version__)

for root, dirs, files in os.walk('./Tensorflow_datasets/data'):
    for file in files:
        print(os.path.join(root, file))