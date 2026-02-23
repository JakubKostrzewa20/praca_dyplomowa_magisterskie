import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

policy = tf.keras.mixed_precision.Policy("mixed_float16")
tf.keras.mixed_precision.set_global_policy(policy)

if len(tf.config.list_physical_devices("GPU")) == 0:
    print("SZKOLISZ BEZ CUDY BEDZIE DLUGO TRWALO")

train_ds = tf.data.Dataset.load("input_data/datasets/100/train_set_100")
test_ds = tf.data.Dataset.load("input_data/datasets/100/test_set_100")
val_ds = tf.data.Dataset.load("input_data/datasets/100/val_set_100")
EPOCHS = 50
BATCH = 16
IMG_SHAPE = (224, 224, 3)

data_augmentation = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomContrast(0.2),
        layers.RandomZoom(0.2),
    ]
)

base_model = tf.keras.applications.MobileNetV3Small(
    input_shape=IMG_SHAPE, include_top=False, weights="imagenet"
)

inputs = tf.keras.Input(shape=IMG_SHAPE)
callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)
model = tf.keras.models.Sequential(
    [
        inputs,
        data_augmentation,
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(38, dtype="float32"),
    ]
)

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    batch_size=BATCH,
    callbacks=callback,
)

acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]

loss = history.history["loss"]
val_loss = history.history["val_loss"]

fig, axes = plt.subplots(2, 1, figsize=(8, 10))

axes[0].plot(acc, label="Training Accuracy")
axes[0].plot(val_acc, label="Validation Accuracy")
axes[0].set_ylabel("Accuracy")
axes[0].set_ylim([0, 1])
axes[0].set_title("Accuracy")
axes[0].legend(loc="lower right")

axes[1].plot(loss, label="Training Loss")
axes[1].plot(val_loss, label="Validation Loss")
axes[1].set_ylabel("Cross Entropy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylim([0, 1.0])
axes[1].set_title("Loss")
axes[1].legend(loc="upper right")

fig.suptitle(
    "Training and Validation Metrics — MobileNetV2 with 100% of data", fontsize=16
)

plt.tight_layout()
plt.subplots_adjust(top=0.93)

plt.savefig(
    "output/plots/training_metrics_mobilenetv2_100.png", dpi=300, bbox_inches="tight"
)

model.save("output/mobilenet/100/mobilenetv2_100.keras")

results = model.evaluate(test_ds, batch_size=16)
