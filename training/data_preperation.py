import tensorflow as tf
import keras
from keras import layers
from sklearn.model_selection import train_test_split
import os
from glob import glob

DATA_DIRECTORY = "input_data/color"
IMG_SIZE = (224, 224)
BATCH = 16
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIRECTORY,
    subset="training",
    validation_split=0.2,
    seed=123,
    image_size=(224, 224),
    batch_size=BATCH,
)
print("train_ds zrobiony")
temp_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIRECTORY,
    subset="validation",
    validation_split=0.2,
    seed=123,
    image_size=(224, 224),
    batch_size=BATCH,
)
temp_count = tf.data.experimental.cardinality(temp_ds)
print("temp_ds zrobiony")
test_ds = temp_ds.take(temp_count // 2)
print("test_ds zrobiony")
val_ds = temp_ds.skip(temp_count // 2)
print("val_ds zrobiony")

num_samples_train = train_ds.cardinality().numpy()
print("Liczba próbek w train_set:", num_samples_train)

num_samples_test = test_ds.cardinality().numpy()
print("Liczba próbek w test_set:", num_samples_test)

num_samples_val = val_ds.cardinality().numpy()
print("Liczba próbek w val_set:", num_samples_val)

test_ds = test_ds.prefetch(tf.data.AUTOTUNE)
trian_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
print("zapis:")

print("test")
test_ds.save("input_data/datasets/test_set")
print("val")
val_ds.save("input_data/datasets/val_set")
print("train")
train_ds.save("input_data/datasets/train_set")
