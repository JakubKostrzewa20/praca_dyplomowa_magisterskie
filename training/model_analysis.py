import tensorflow as tf
import keras
from keras import layers
import tensorflow as tf
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

test_ds = tf.data.Dataset.load("input_data/datasets/test_set")
EPOCHS = 50
BATCH = 16
IMG_SHAPE = (224, 224, 3)

model = tf.keras.models.load_model("output/mobilenet/mobile_net_v2.keras", compile=True)
y_true = []
y_pred = []
for x, y in test_ds:
    logits = model(x, training=False)
    y_pred.extend(tf.argmax(logits, axis=1).numpy())
    y_true.extend(y.numpy())

y_pred_labels = np.array(y_pred)
y_true_labels = np.array(y_true)
print(classification_report(y_true_labels, y_pred_labels))

print(
    "Accuracy: ",
    accuracy_score(
        y_true_labels,
        y_pred_labels,
    ),
)
print("Recall: ", recall_score(y_true_labels, y_pred_labels, average="micro"))
print("Precision: ", precision_score(y_true_labels, y_pred_labels, average="micro"))
print("Macro F1-Score: ", f1_score(y_true_labels, y_pred_labels, average="macro"))
print("Micro F1-Score: ", f1_score(y_true_labels, y_pred_labels, average="micro"))
