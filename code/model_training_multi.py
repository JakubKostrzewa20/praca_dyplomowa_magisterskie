import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

policy = tf.keras.mixed_precision.Policy("mixed_float16")
tf.keras.mixed_precision.set_global_policy(policy)

if len(tf.config.list_physical_devices("GPU")) == 0:
    print("NO GPU DETECTED")


sets=[25,50,100]
for s in sets:

    train_ds = tf.data.Dataset.load(f"input_data/datasets/{s}/train_set_{s}")
    test_ds = tf.data.Dataset.load(f"input_data/datasets/{s}/test_set_{s}")
    val_ds = tf.data.Dataset.load(f"input_data/datasets/{s}/val_set_{s}")

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

    base_model = tf.keras.applications.ResNet50V2(
        input_shape=IMG_SHAPE, include_top=False, weights="imagenet"
    )

    inputs = tf.keras.Input(shape=IMG_SHAPE)
    callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights = True)
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
        f"Training and Validation Metrics — ResNet50V2 with {s}% of data", fontsize=16
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)

    plt.savefig(
        f"output/plots/training_metrics_resnet50V2_{s}_new.png", dpi=300, bbox_inches="tight"
    )

    model.save(f"output/resnet/{s}/resnet50V2_{s}_new.keras")

    results = model.evaluate(test_ds, batch_size=16)
