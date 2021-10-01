# word embeddings
#####################################################################
# This tutorial contains an introduction to word embeddings. 
# You will train your own word embeddings using a simple Keras 
# model for a sentiment classification task, and then visualize 
# them in the Embedding Projector (shown in the image below). 
# https://www.tensorflow.org/text/guide/word_embeddings

# Setup
import io
import os
import re
import shutil
import string
import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
# made corrections to the bad path calling TextVectorization
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization


if os.path.join(os.getcwd(), 'aclImdb') == "":
    url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

    dataset = tf.keras.utils.get_file("aclImdb_v1.tar.gz", url,
                                    untar=True, cache_dir='.',
                                    cache_subdir='')

    dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')

#################################################################

dataset_dir = os.path.join(os.getcwd(), 'aclImdb')
os.listdir(dataset_dir)
# Take a look at the train/ directory
train_dir = os.path.join(dataset_dir, 'train')
os.listdir(train_dir)
# removing some additional folders
if os.path.exists(os.getcwd()+"/aclImdb/unsup") == True:
    remove_dir = os.path.join(train_dir, 'unsup')
    shutil.rmtree(remove_dir)
    print("Removing additonal files...")
# create a tf.data.Dataset using tf.keras.preprocessing.text_dataset_from_directory. 
batch_size = 1024
seed = 123
train_ds = tf.keras.preprocessing.text_dataset_from_directory(
    'aclImdb/train', batch_size=batch_size, validation_split=0.2,
    subset='training', seed=seed)
val_ds = tf.keras.preprocessing.text_dataset_from_directory(
    'aclImdb/train', batch_size=batch_size, validation_split=0.2,
    subset='validation', seed=seed)

# Take a look at a few movie reviews and their labels (1: positive, 0: negative) from the train dataset.
for text_batch, label_batch in train_ds.take(1):
  for i in range(5):
    print(label_batch[i].numpy(), text_batch.numpy()[i])

# Configure the dataset for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Using the Embedding layer
# Embed a 1,000 word vocabulary into 5 dimensions.
embedding_layer = tf.keras.layers.Embedding(1000, 5)

# When you create an Embedding layer, the weights for the embedding are randomly initialized (just like any other layer). 
result = embedding_layer(tf.constant([1, 2, 3]))
result.numpy()

# For text or sequence problems, the Embedding layer takes a 2D tensor of integers, of shape (samples, sequence_length), where each entry is a sequence of integers. 
result = embedding_layer(tf.constant([[0, 1, 2], [3, 4, 5]]))
result.shape

# define the dataset preprocessing steps required for your sentiment classification model. 

# Create a custom standardization function to strip HTML break tags '<br />'.
def custom_standardization(input_data):
  lowercase = tf.strings.lower(input_data)
  stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
  return tf.strings.regex_replace(stripped_html,
                                  '[%s]' % re.escape(string.punctuation), '')


# Vocabulary size and number of words in a sequence.
vocab_size = 10000
sequence_length = 100

# Use the text vectorization layer to normalize, split, and map strings to
# integers. Note that the layer uses the custom standardization defined above.
# Set maximum_sequence length as all samples are not of the same length.
vectorize_layer = TextVectorization(
    standardize=custom_standardization,
    max_tokens=vocab_size,
    output_mode='int',
    output_sequence_length=sequence_length)

# Make a text-only dataset (no labels) and call adapt to build the vocabulary.
text_ds = train_ds.map(lambda x, y: x)
vectorize_layer.adapt(text_ds)


# Create a classification model
embedding_dim=16

model = Sequential([
  vectorize_layer,
  Embedding(vocab_size, embedding_dim, name="embedding"),
  GlobalAveragePooling1D(),
  Dense(16, activation='relu'),
  Dense(1)
])

# Compile and train the model
# You will use TensorBoard to visualize metrics including loss and accuracy. Create a tf.keras.callbacks.TensorBoard.
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")

# Compile and train the model using the Adam optimizer and BinaryCrossentropy loss.
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    callbacks=[tensorboard_callback])

# You can look into the model summary to learn more about each layer of the model.
model.summary()


# Visualize the model metrics in TensorBoard.

# Retrieve the trained word embeddings and save them to disk
weights = model.get_layer('embedding').get_weights()[0]
vocab = vectorize_layer.get_vocabulary()

# Write the weights to disk. 
out_v = io.open('vectors.tsv', 'w', encoding='utf-8')
out_m = io.open('metadata.tsv', 'w', encoding='utf-8')

for index, word in enumerate(vocab):
  if index == 0:
    continue  # skip 0, it's padding.
  vec = weights[index]
  out_v.write('\t'.join([str(x) for x in vec]) + "\n")
  out_m.write(word + "\n")
out_v.close()
out_m.close()

print("Word Embedding completed...")
# END OF TUTORIAL