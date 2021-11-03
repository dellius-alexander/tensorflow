# Setup Datasets

### Module [Tensorflow Dataset (tfds)](https://www.tensorflow.org/datasets/api_docs/python/tfds):

***Note: these documentation can be found on [https://www.tensorflow.org/datasets](https://www.tensorflow.org/datasets).***

- `tensorflow_datasets (tfds)` defines a collection of datasets ready-to-use with TensorFlow.

- Each dataset is defined as a tfds.core.DatasetBuilder, which encapsulates the logic to download the dataset and construct an input pipeline, as well as contains the dataset documentation (version, splits, number of examples, etc.).
- The main library entrypoints are:

    - `tfds.builder`: fetch a tfds.core.DatasetBuilder by name
    - `tfds.load`: convenience method to construct a builder, download the data, and create an input pipeline, returning a tf.data.Dataset.

- `tfds.core.DatasetBuilder` is the Abstract base class for all datasets.
    - DatasetBuilder has 3 key methods:

    `DatasetBuilder.info`: documents the dataset, including feature names, types, and shapes, version, splits, citation, etc.
    `DatasetBuilder.download_and_prepare`: downloads the source data and writes it to disk.
    `DatasetBuilder.as_dataset`: builds an input pipeline using tf.data.Datasets.

Configuration: Some DatasetBuilders expose multiple variants of the dataset by defining a tfds.core.BuilderConfig subclass and accepting a config object (or name) on construction. Configurable datasets expose a pre-defined set of configurations in DatasetBuilder.builder_configs. 

```python
# DatasetBuilder Constructor
tfds.core.DatasetBuilder(
    *,
    data_dir: Optional[utils.PathLike] = None,
    config: Union[None, str, tfds.core.BuilderConfig] = None,
    version: Union[None, str, tfds.core.Version] = None
)

# Typical DatasetBuilder usage:

mnist_builder = tfds.builder("open_images_v4")
mnist_info = mnist_builder.info
mnist_builder.download_and_prepare()
datasets = mnist_builder.as_dataset()

# Visualize your dataset with fiftyone before we do anything to verify
#     your model input.
session = fo.launch_app(dataset)
session.wait()

# setup and check dataset dataType with assert instance()
train_dataset, test_dataset = datasets["train"], datasets["test"]
assert isinstance(train_dataset, tf.data.Dataset)

# And then the rest of your input pipeline
train_dataset = train_dataset.repeat().shuffle(1024).batch(128)
train_dataset = train_dataset.prefetch(2)
features = tf.compat.v1.data.make_one_shot_iterator(train_dataset).get_next()
image, label = features['image'], features['label']

```
