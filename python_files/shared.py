import os
import argparse
from datetime import datetime
import random
import json

from callbacks import make_epoch_log_callback


IMAGE_SIZE = (180, 180)


def setup(results_dir):
    """Heavy imports + output directory creation. Returns ctx dict."""
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

    return {
        "np": np, "keras": keras, "layers": layers,
        "tf_data": tf_data, "plt": plt,
        "img_dir": img_dir, "model_dir": model_dir,
    }


def load_and_visualize(ctx, source_dir, validation_split, seed, batch_size):
    """Load dataset, save sample/augmented images, apply augmentation + prefetch.
    Returns (train_ds, val_ds, class_names, num_classes, data_augmentation)."""
    np = ctx["np"]
    keras = ctx["keras"]
    layers = ctx["layers"]
    tf_data = ctx["tf_data"]
    plt = ctx["plt"]
    img_dir = ctx["img_dir"]

    train_ds, val_ds = keras.utils.image_dataset_from_directory(
        source_dir,
        validation_split=validation_split,
        subset="both",
        seed=seed,
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)

    # Sample images
    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(np.array(images[i]).astype("uint8"))
            plt.title(class_names[int(labels[i])], fontsize=7)
            plt.axis("off")
    plt.savefig(os.path.join(img_dir, "sample_images.png"))
    plt.close()

    # Data augmentation
    data_augmentation_layers = [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
    ]

    def data_augmentation(images):
        for layer in data_augmentation_layers:
            images = layer(images)
        return images

    # Augmented images
    plt.figure(figsize=(10, 10))
    for images, _ in train_ds.take(1):
        for i in range(9):
            augmented_images = data_augmentation(images)
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(np.array(augmented_images[0]).astype("uint8"))
            plt.axis("off")
    plt.savefig(os.path.join(img_dir, "augmented_images.png"))
    plt.close()

    # Performance config
    train_ds = train_ds.map(
        lambda img, label: (data_augmentation(img), label),
        num_parallel_calls=tf_data.AUTOTUNE,
    )
    train_ds = train_ds.prefetch(tf_data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf_data.AUTOTUNE)

    return train_ds, val_ds, class_names, num_classes, data_augmentation


def make_model(ctx, num_classes, data_augmentation=None):
    """Build the CNN. If data_augmentation is provided, applies it in the entry block."""
    keras = ctx["keras"]
    layers = ctx["layers"]
    img_dir = ctx["img_dir"]

    inputs = keras.Input(shape=IMAGE_SIZE + (3,))

    # Entry block
    x = inputs
    if data_augmentation is not None:
        x = data_augmentation(x)
    x = layers.Rescaling(1.0 / 255)(x)
    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x

    for size in [256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])
        previous_block_activation = x

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    units = 1 if num_classes == 2 else num_classes

    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(units, activation=None)(x)
    model = keras.Model(inputs, outputs)

    keras.utils.plot_model(model, to_file=os.path.join(img_dir, "model_architecture.png"), show_shapes=True)
    return model


def train(ctx, model, train_ds, val_ds, epochs, learning_rate, num_classes, results_dir):
    """Compile and train model. Selects binary vs categorical loss based on num_classes."""
    keras = ctx["keras"]
    plt = ctx["plt"]
    img_dir = ctx["img_dir"]
    model_dir = ctx["model_dir"]

    callbacks = [
        make_epoch_log_callback(
            keras=keras,
            plt=plt,
            log_path=os.path.join(results_dir, "epochs.json"),
            convergence_dir=img_dir,
            model_dir=model_dir,
        ),
    ]

    if num_classes == 2:
        loss = keras.losses.BinaryCrossentropy(from_logits=True)
        metrics = [keras.metrics.BinaryAccuracy(name="acc")]
    else:
        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metrics = [keras.metrics.SparseCategoricalAccuracy(name="acc")]

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=loss,
        metrics=metrics,
    )
    model.fit(
        train_ds,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=val_ds,
    )
    model.save(os.path.join(model_dir, "model_final.keras"))


def evaluate(ctx, model, val_ds, class_names, num_classes):
    """Generate confusion matrix and sample predictions plots."""
    np = ctx["np"]
    keras = ctx["keras"]
    plt = ctx["plt"]
    img_dir = ctx["img_dir"]

    # Collect predictions
    all_labels = []
    all_preds = []
    all_probs = []
    for images, labels in val_ds:
        logits = model.predict(images, verbose=0)
        if num_classes == 2:
            scores = np.array(keras.ops.sigmoid(logits[:, 0]))
            preds = (scores >= 0.5).astype(int)
            probs = np.column_stack([1 - scores, scores])
        else:
            probs = np.array(keras.ops.softmax(logits))
            preds = np.argmax(probs, axis=1)
        all_labels.extend(np.array(labels).tolist())
        all_preds.extend(preds.tolist())
        all_probs.extend(probs.tolist())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    # Confusion matrix
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(all_labels, all_preds):
        cm[int(true)][int(pred)] += 1

    fig_size = max(5, num_classes // 6)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    label_fontsize = 9 if num_classes <= 10 else 6
    ax.set_xticklabels(class_names, rotation=90, fontsize=label_fontsize)
    ax.set_yticklabels(class_names, fontsize=label_fontsize)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (validation)")
    if num_classes <= 10:
        for i in range(num_classes):
            for j in range(num_classes):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(os.path.join(img_dir, "confusion_matrix.png"), dpi=150)
    plt.close(fig)

    # Sample predictions grid
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    axes = axes.flatten()
    shown = 0
    for images, labels in val_ds:
        if shown >= 9:
            break
        logits = model.predict(images, verbose=0)
        if num_classes == 2:
            scores = np.array(keras.ops.sigmoid(logits[:, 0]))
            pred_classes = (scores >= 0.5).astype(int)
            confidences = np.where(scores >= 0.5, scores, 1 - scores)
        else:
            probs = np.array(keras.ops.softmax(logits))
            pred_classes = np.argmax(probs, axis=1)
            confidences = np.max(probs, axis=1)
        for i in range(len(images)):
            if shown >= 9:
                break
            true_label = class_names[int(labels[i])]
            pred_label = class_names[pred_classes[i]]
            correct = true_label == pred_label
            axes[shown].imshow(np.array(images[i]).astype("uint8"))
            axes[shown].set_title(
                f"True: {true_label}\nPred: {pred_label} ({confidences[i]:.0%})",
                color="green" if correct else "red", fontsize=7
            )
            axes[shown].axis("off")
            shown += 1
    fig.tight_layout()
    fig.savefig(os.path.join(img_dir, "sample_predictions.png"))
    plt.close(fig)


# --- CLI helpers ---

def _positive_int(value):
    v = int(value)
    if v <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {value}")
    return v


def _positive_perc(value):
    v = float(value)
    if not (0.0 < v < 1.0):
        raise argparse.ArgumentTypeError(f"must be in (0, 1), got {value}")
    return v


def make_parser(description, default_source_dir):
    """Create an ArgumentParser with all shared training arguments."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-s", "--source-dir", default=default_source_dir,
                        help=f"Path to the image dataset directory (default: {default_source_dir})")
    parser.add_argument("-o", "--output-dir", default=None,
                        help="Directory for run outputs (default: data/results/YYYY-MM-DD HHMMSS)")
    parser.add_argument("-e", "--epochs", type=_positive_int, default=50,
                        help="Number of training epochs, must be > 0 (default: 50)")
    parser.add_argument("-l", "--learning-rate", type=_positive_perc, default=0.0001,
                        help="Adam learning rate, must be in (0, 1) (default: 0.0001)")
    parser.add_argument("-b", "--batch-size", type=_positive_int, default=50,
                        help="Training batch size, must be > 0 (default: 50)")
    parser.add_argument("-v", "--validation-split", type=_positive_perc, default=0.2,
                        help="Fraction of data to use for validation, must be in (0, 1) (default: 0.2)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for dataset split, None means random (default: None)")
    return parser


def resolve_args(parser):
    """Parse args, validate, resolve defaults, save args.json. Returns args namespace."""
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

    return args
