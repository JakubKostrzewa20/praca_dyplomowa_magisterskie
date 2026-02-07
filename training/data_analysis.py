import tensorflow as tf
import keras
from keras import layers
import numpy as np
import os
import matplotlib.pyplot as plt

BASE_DIR = "input_data/color"
class_count = {}
for cls in sorted(os.listdir(BASE_DIR)):
    cls_path = os.path.join(BASE_DIR, cls)
    class_count[cls] = len([f for f in os.listdir(cls_path)])

print("Liczba klas:", len(class_count))
print("Statystyki dla konkretnych klas:", class_count)

sorted_class_count = dict(sorted(class_count.items(), key=lambda item: item[1]))
print("Klasy posortowane dla ilości zdjęć: ", sorted_class_count)

train_ds = tf.data.Dataset.load("input_data/datasets/train_set")
test_ds = tf.data.Dataset.load("input_data/datasets/test_set")
val_ds = tf.data.Dataset.load("input_data/datasets/val_set")

num_samples_train = train_ds.cardinality().numpy()
print("Liczba próbek w train_set:", num_samples_train)

num_samples_test = test_ds.cardinality().numpy()
print("Liczba próbek w test_set:", num_samples_test)

num_samples_val = val_ds.cardinality().numpy()
print("Liczba próbek w val_set:", num_samples_val)

counts = np.array(list(class_count.values()))
print("Najmniejsza ilość zdjęć dla klasy: ", counts.min())
print("Największa ilość zdjęć dla klasy: ", counts.max())
print("Odchylenie standardowe: ", counts.std())
print("Średnia zdjęć na klasę: ", counts.mean())

data_augmentation = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomContrast(0.2),
        layers.RandomZoom(0.2),
    ]
)

for batch, _ in train_ds.take(1):
    plt.figure(figsize=(10, 10))
    image = batch[0]
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        augmented_image = data_augmentation(tf.expand_dims(image, 0), training=True)
        plt.imshow(tf.cast(augmented_image[0], tf.uint8))
        plt.axis("off")
plt.show()
plt.savefig("output/plots/wykres.jpg")


labels = set()
for _, y in test_ds:
    labels.update(y.numpy())

print("Numery klas w dataset:", sorted(labels))
