import tensorflow as tf
import keras
from keras import layers

DATA_DIRECTORY = "input_data/color"


train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIRECTORY,
    subset="training",
    validation_split=0.2,
    seed=123,
    image_size=(224, 224),
    batch_size=None,
)
test_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIRECTORY,
    subset="validation",
    validation_split=0.2,
    seed=123,
    image_size=(224, 224),
    batch_size=None,
)

train_ds = train_ds.shuffle(buffer_size=50000, seed=123, reshuffle_each_iteration=False)

train_ds.save("input_data/datasets/train_set")
test_ds.save("input_data/datasets/test_set")
