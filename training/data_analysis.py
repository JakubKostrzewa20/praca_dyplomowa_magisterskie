import tensorflow as tf
import keras
from keras import layers
import numpy as np
import os

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

num_samples_train = train_ds.cardinality().numpy()
print("Liczba próbek w train_set:", num_samples_train)

num_samples_test = test_ds.cardinality().numpy()
print("Liczba próbek w test_set:", num_samples_test)

counts = np.array(list(class_count.values()))
print("Najmniejsza ilość zdjęć dla klasy: ", counts.min())
print("Największa ilość zdjęć dla klasy: ", counts.max())
print("Odchylenie standardowe: ", counts.std())
print("Średnia zdjęć na klasę: ", counts.mean())
