import tensorflow as tf
import keras
from keras import layers

train_ds = tf.data.Dataset.load("input_data/datasets/train_set")
test_ds = tf.data.Dataset.load("input_data/datasets/test_set")
