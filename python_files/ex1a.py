import os
import argparse
import numpy as np
import keras
from keras import layers
from tensorflow import data as tf_data
import matplotlib.pyplot as plt
from datetime import datetime


def run(source_dir, results_base, epochs, learning_rate):
    run_ts = datetime.now().strftime("%Y-%m-%d %H%M%S")
    results_dir = os.path.join(results_base, run_ts)
    img_dir = os.path.join(results_dir, "img")
    model_dir = os.path.join(results_dir, "model")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Filter out corrupted images
    num_skipped = 0
    for folder_name in ("Cat", "Dog"):
        folder_path = os.path.join(source_dir, folder_name)
        for fname in os.listdir(folder_path):
            fpath = os.path.join(folder_path, fname)
            try:
                fobj = open(fpath, "rb")
                is_jfif = b"JFIF" in fobj.peek(10)
            finally:
                fobj.close()

            if not is_jfif:
                num_skipped += 1
                os.remove(fpath)

    print(f"Deleted {num_skipped} images.")

    image_size = (180, 180)
    batch_size = 128

    # Generate a Dataset
    train_ds, val_ds = keras.utils.image_dataset_from_directory(
        source_dir,
        validation_split=0.2,
        subset="both",
        seed=1337,
        image_size=image_size,
        batch_size=batch_size,
    )

    # Visualize the data
    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(np.array(images[i]).astype("uint8"))
            plt.title(int(labels[i]))
            plt.axis("off")
    plt.savefig(os.path.join(img_dir, "sample_images.png"))
    plt.close()

    # Using image data augmentation
    data_augmentation_layers = [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
    ]

    def data_augmentation(images):
        for layer in data_augmentation_layers:
            images = layer(images)
        return images

    plt.figure(figsize=(10, 10))
    for images, _ in train_ds.take(1):
        for i in range(9):
            augmented_images = data_augmentation(images)
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(np.array(augmented_images[0]).astype("uint8"))
            plt.axis("off")
    plt.savefig(os.path.join(img_dir, "augmented_images.png"))
    plt.close()

    # Configure the dataset for performance
    train_ds = train_ds.map(
        lambda img, label: (data_augmentation(img), label),
        num_parallel_calls=tf_data.AUTOTUNE,
    )
    train_ds = train_ds.prefetch(tf_data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf_data.AUTOTUNE)

    # Build a model
    def make_model(input_shape, num_classes):
        inputs = keras.Input(shape=input_shape)

        # Entry block
        x = data_augmentation(inputs)
        x = layers.Rescaling(1.0 / 255)(x)
        x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        previous_block_activation = x  # Set aside residual

        for size in [256, 512, 728]:
            x = layers.Activation("relu")(x)
            x = layers.SeparableConv2D(size, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.Activation("relu")(x)
            x = layers.SeparableConv2D(size, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

            # Project residual
            residual = layers.Conv2D(size, 1, strides=2, padding="same")(
                previous_block_activation
            )
            x = layers.add([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual

        x = layers.SeparableConv2D(1024, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        x = layers.GlobalAveragePooling2D()(x)
        units = 1 if num_classes == 2 else num_classes

        x = layers.Dropout(0.25)(x)
        # We specify activation=None so as to return logits
        outputs = layers.Dense(units, activation=None)(x)
        return keras.Model(inputs, outputs)

    model = make_model(input_shape=image_size + (3,), num_classes=2)
    keras.utils.plot_model(model, to_file=os.path.join(model_dir, "model_architecture.png"), show_shapes=True)

    # Train the model
    callbacks = [
        keras.callbacks.ModelCheckpoint(os.path.join(model_dir, "save_at_{epoch}.keras")),
    ]
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[keras.metrics.BinaryAccuracy(name="acc")],
    )
    history = model.fit(
        train_ds,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=val_ds,
    )

    # Convergence plot
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["acc"], label="Train accuracy")
    plt.plot(history.history["val_acc"], label="Val accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy convergence")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train loss")
    plt.plot(history.history["val_loss"], label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss convergence")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "convergence.png"))
    plt.close()

    # Run inference on new data
    sample_img_path = os.path.join(source_dir, "Cat", "6779.jpg")
    img = keras.utils.load_img(sample_img_path, target_size=image_size)
    plt.figure()
    plt.imshow(img)
    plt.savefig(os.path.join(img_dir, "inference_sample.png"))
    plt.close()

    img_array = keras.utils.img_to_array(img)
    img_array = keras.ops.expand_dims(img_array, 0)  # Create batch axis

    predictions = model.predict(img_array)
    score = float(keras.ops.sigmoid(predictions[0][0]))
    print(f"This image is {100 * (1 - score):.2f}% cat and {100 * score:.2f}% dog.")
    print(f"Results saved to: {results_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a cat/dog classifier")
    parser.add_argument("--source-dir", default="cat_and_dog_images",
                        help="Path to the image dataset directory (default: cat_and_dog_images)")
    parser.add_argument("--results-dir", default=os.path.join("data", "results"),
                        help="Base directory for run outputs (default: data/results)")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of training epochs (default: 1)")
    parser.add_argument("--learning-rate", type=float, default=0.0001,
                        help="Adam learning rate (default: 0.0001)")
    args = parser.parse_args()

    run(
        source_dir=args.source_dir,
        results_base=args.results_dir,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )
