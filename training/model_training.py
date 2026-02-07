import tensorflow as tf
from keras import layers
import tensorflow as tf

if len(tf.config.list_physical_devices("GPU")) == 0:
    print("SZKOLISZ BEZ CUDY BEDZIE DLUGO TRWALO")

train_ds = tf.data.Dataset.load("input_data/datasets/train_set")
test_ds = tf.data.Dataset.load("input_data/datasets/test_set")
val_ds = tf.data.Dataset.load("input_data/datasets/val_set")
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

preprocess_input = tf.keras.applications.mobilenet_v3.preprocess_input

base_model = tf.keras.applications.MobileNetV3Small(
    input_shape=IMG_SHAPE, include_top=False, weights="imagenet"
)

inputs = tf.keras.Input(shape=IMG_SHAPE)
callback = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=5)
model = tf.keras.models.Sequential(
    [
        inputs,
        data_augmentation,
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(38),
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

model.save("output/mobilenet/mobile_net_v3small.keras")

results = model.evaluate(test_ds, batch_size=16)
