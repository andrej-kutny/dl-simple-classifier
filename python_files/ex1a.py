import os
import argparse
from datetime import datetime
import random
import json
import time

def run(source_dir, results_dir, epochs, learning_rate, batch_size, validation_split, seed):
    import numpy as np
    import keras
    from keras import layers
    from tensorflow import data as tf_data
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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

    # Generate a Dataset
    train_ds, val_ds = keras.utils.image_dataset_from_directory(
        source_dir,
        validation_split=validation_split,
        subset="both",
        seed=seed,
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
    keras.utils.plot_model(model, to_file=os.path.join(img_dir, "model_architecture.png"), show_shapes=True)

    class EpochLogCallback(keras.callbacks.Callback):
        def __init__(self, log_path, convergence_path):
            super().__init__()
            self._log_path = log_path
            self._convergence_path = convergence_path
            self._epoch_start_ts = None
            self._epoch_start_mono = None
            with open(self._log_path, "w") as f:
                json.dump({}, f)

        def on_epoch_begin(self, epoch, logs=None):
            self._epoch_start_ts = datetime.now().isoformat()
            self._epoch_start_mono = time.monotonic()

        def on_epoch_end(self, epoch, logs=None):
            elapsed = time.monotonic() - self._epoch_start_mono

            # Update epochs.json
            with open(self._log_path, "r+") as f:
                data = json.load(f)
                data[epoch + 1] = {
                    "start": self._epoch_start_ts,
                    "end": datetime.now().isoformat(),
                    "elapsed_seconds": round(elapsed, 3),
                    "vals": {
                        "train_loss": logs.get("loss"),
                        "val_loss": logs.get("val_loss"),
                        "train_acc": logs.get("acc"),
                        "val_acc": logs.get("val_acc"),
                    },
                }
                f.seek(0)
                json.dump(data, f, indent=2)
                f.truncate()

            # Update convergence plot from the same data
            epochs_range = list(data.keys())
            train_acc = [data[e]["vals"]["train_acc"] for e in epochs_range]
            val_acc   = [data[e]["vals"]["val_acc"]   for e in epochs_range]
            train_loss = [data[e]["vals"]["train_loss"] for e in epochs_range]
            val_loss   = [data[e]["vals"]["val_loss"]   for e in epochs_range]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            ax1.plot(epochs_range, train_acc, color="lightgreen", label="Train accuracy")
            ax1.plot(epochs_range, val_acc,   color="green",      label="Val accuracy")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Accuracy")
            ax1.set_title("Accuracy convergence")
            ax1.legend()

            ax2.plot(epochs_range, train_loss, color="lightcoral", label="Train loss")
            ax2.plot(epochs_range, val_loss,   color="red",        label="Val loss")
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Loss")
            ax2.set_title("Loss convergence")
            ax2.legend()

            fig.tight_layout()
            fig.savefig(self._convergence_path)
            plt.close(fig)

    # Train the model
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(model_dir, "checkpoint_{epoch}.keras"),
            monitor='val_accuracy',
            mode='max',
            save_best_only=True),
        EpochLogCallback(
            log_path=os.path.join(results_dir, "epochs.json"),
            convergence_path=os.path.join(img_dir, "convergence.png"),
        ),
    ]
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[keras.metrics.BinaryAccuracy(name="acc")],
    )
    model.fit(
        train_ds,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=val_ds,
    )

    # Confusion matrix on validation set
    all_labels = []
    all_preds = []
    for images, labels in val_ds:
        logits = model.predict(images, verbose=0)
        scores = keras.ops.sigmoid(logits[:, 0])
        preds = (np.array(scores) >= 0.5).astype(int)
        all_labels.extend(np.array(labels).tolist())
        all_preds.extend(preds.tolist())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    class_names = ["Cat", "Dog"]
    cm = np.zeros((2, 2), dtype=int)
    for true, pred in zip(all_labels, all_preds):
        cm[int(true)][int(pred)] += 1

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (validation)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(os.path.join(img_dir, "confusion_matrix.png"))
    plt.close(fig)

    # Sample predictions grid on validation set
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    axes = axes.flatten()
    shown = 0
    for images, labels in val_ds:
        if shown >= 9:
            break
        logits = model.predict(images, verbose=0)
        scores = keras.ops.sigmoid(logits[:, 0])
        for i in range(len(images)):
            if shown >= 9:
                break
            score = float(scores[i])
            true_label = class_names[int(labels[i])]
            pred_label = class_names[int(score >= 0.5)]
            confidence = score if score >= 0.5 else 1 - score
            correct = true_label == pred_label
            axes[shown].imshow(np.array(images[i]).astype("uint8"))
            axes[shown].set_title(
                f"True: {true_label}\nPred: {pred_label} ({confidence:.0%})",
                color="green" if correct else "red", fontsize=9
            )
            axes[shown].axis("off")
            shown += 1
    fig.tight_layout()
    fig.savefig(os.path.join(img_dir, "sample_predictions.png"))
    plt.close(fig)

    print(f"Results saved to: {results_dir}")


def positive_int(value):
    v = int(value)
    if v <= 0:
        raise argparse.ArgumentTypeError(f"epochs must be a positive integer, got {value}")
    return v


def positive_perc(value):
    v = float(value)
    if not (0.0 < v < 1.0):
        raise argparse.ArgumentTypeError(f"learning-rate must be in (0, 1), got {value}")
    return v


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a cat/dog classifier")
    parser.add_argument("-s", "--source-dir", default="cat_and_dog_images",
                        help="Path to the image dataset directory (default: cat_and_dog_images)")
    parser.add_argument("-o", "--output-dir", default=None,
                        help="Directory for run outputs (default: data/results/YYYY-MM-DD HHMMSS)")
    parser.add_argument("-e", "--epochs", type=positive_int, default=50,
                        help="Number of training epochs, must be > 0 (default: 50)")
    parser.add_argument("-l", "--learning-rate", type=positive_perc, default=0.0001,
                        help="Adam learning rate, must be in (0, 1) (default: 0.0001)")
    parser.add_argument("-b", "--batch-size", type=positive_int, default=50,
                        help="Training batch size, must be > 0 (default: 50)")
    parser.add_argument("-v", "--validation-split", type=positive_perc, default=0.2,
                        help="Fraction of data to use for validation, must be in (0, 1) (default: 0.2)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for dataset split, None means random (default: None)")
    args = parser.parse_args()

    if not os.path.isdir(args.source_dir):
        parser.error(f"--source-dir does not exist or is not a directory: {args.source_dir}")

    if args.seed is None:
        args.seed = random.randint(0, 2**31 - 1)
        print(f"Using random seed: {args.seed}")
    
    if args.output_dir is None:
        args.output_dir = os.path.join(
            "data", "results", datetime.now().strftime("%Y-%m-%d %H%M%S")
        )
        print(f"Using output directory: {args.output_dir}")

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(args.__dict__, f, indent=2)

    run(
        source_dir=args.source_dir,
        results_dir=args.output_dir,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        seed=args.seed,
    )
