import tensorflow as tf
import keras
from keras import layers

DATA_DIRECTORY = "input_data/new_datasets/100/train"
IMG_SIZE = (224, 224)
BATCH = 16

train_ds = tf.keras.utils.image_dataset_from_directory(
    directory = "input_data/new_datasets/25/train",
    image_size=IMG_SIZE,
    batch_size=BATCH,
)
print("train_ds stworzony")
val_ds = tf.keras.utils.image_dataset_from_directory(
    directory = "input_data/new_datasets/25/val",
    image_size=IMG_SIZE,
    batch_size=BATCH,
)
test_ds=tf.keras.utils.image_dataset_from_directory(
    directory = "input_data/new_datasets/25/test",
    image_size=IMG_SIZE,
    batch_size=BATCH,
)

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
test_ds.save("input_data/datasets/25/test_set_25")
print("val_ds")
val_ds.save("input_data/datasets/25/val_set_25")
print("train_ds")
train_ds.save("input_data/datasets/25/train_set_25")