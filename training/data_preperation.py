import tensorflow as tf
import keras
from keras import layers

DATA_DIRECTORY = "input_data/color"
IMG_SIZE = (224, 224)
BATCH = 16

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIRECTORY,
    subset="training",
    validation_split=0.2,
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH,
)
print("train_ds stworzony")
temp_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIRECTORY,
    subset="validation",
    validation_split=0.2,
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH,
)
temp_count = tf.data.experimental.cardinality(temp_ds)
print("temp_ds stworzony")
test_ds = temp_ds.take(temp_count // 2)
print("test_ds stworzony")
val_ds = temp_ds.skip(temp_count // 2)
print("val_ds stworzony")

num_samples_train = train_ds.cardinality().numpy()
print("Liczba próbek w train_set:", num_samples_train)

num_samples_test = test_ds.cardinality().numpy()
print("Liczba próbek w test_set:", num_samples_test)

num_samples_val = val_ds.cardinality().numpy()
print("Liczba próbek w val_set:", num_samples_val)

test_ds = test_ds.prefetch(tf.data.AUTOTUNE)
trian_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

print("Zapisuje:")
print("test_ds")
test_ds.save("input_data/datasets/test_set")
print("val_ds")
val_ds.save("input_data/datasets/val_set")
print("train_ds")
train_ds.save("input_data/datasets/train_set")