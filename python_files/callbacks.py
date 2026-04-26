import os
import json
import time
from datetime import datetime


def parse_epoch_data(data):
    """Extract plotting series from epochs.json data dict.
    Returns (epochs_range, train_acc, val_acc, train_loss, val_loss)."""
    epochs_range, train_acc, val_acc, train_loss, val_loss = [], [], [], [], []
    for e, v in sorted(data.items(), key=lambda x: int(x[0])):
        epochs_range.append(int(e))
        train_acc.append(v["vals"]["train_acc"])
        val_acc.append(v["vals"]["val_acc"])
        train_loss.append(v["vals"]["train_loss"])
        val_loss.append(v["vals"]["val_loss"])
    return epochs_range, train_acc, val_acc, train_loss, val_loss


def plot_accuracy(plt, epochs_range, train_acc, val_acc, best_checkpoints=None,
                  label=None, train_color="lightgreen", val_color="green", fig_ax=None):
    """Generate accuracy convergence plot. Returns (fig, ax).
    Pass fig_ax=(fig, ax) to add series to an existing plot."""
    if fig_ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig, ax = fig_ax

    train_label = "Train accuracy" if label is None else f"{label} train"
    val_label = "Val accuracy" if label is None else f"{label} val"

    ax.plot(epochs_range, train_acc, color=train_color, label=train_label)
    ax.plot(epochs_range, val_acc,   color=val_color,   label=val_label)

    if best_checkpoints:
        best_epochs = [p[0] for p in best_checkpoints]
        best_accs   = [p[1] for p in best_checkpoints]
        ax.plot(best_epochs, best_accs, color=val_color, linewidth=0.8,
                linestyle="--", label="Best checkpoints")
        for ckpt_epoch, ckpt_acc in best_checkpoints:
            ax.plot(ckpt_epoch, ckpt_acc, "o", color=val_color, markersize=5)
            ax.annotate(f"{ckpt_epoch}: {ckpt_acc:.5f}",
                        xy=(ckpt_epoch, ckpt_acc),
                        xytext=(0, 6), textcoords="offset points", rotation=45,
                        fontsize=7, color=val_color)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy convergence")
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_loss(plt, epochs_range, train_loss, val_loss,
              label=None, train_color="lightcoral", val_color="red", fig_ax=None):
    """Generate loss convergence plot. Returns (fig, ax).
    Pass fig_ax=(fig, ax) to add series to an existing plot."""
    if fig_ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig, ax = fig_ax

    train_label = "Train loss" if label is None else f"{label} train"
    val_label = "Val loss" if label is None else f"{label} val"

    ax.plot(epochs_range, train_loss, color=train_color, label=train_label)
    ax.plot(epochs_range, val_loss,   color=val_color,   label=val_label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss convergence")
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_convergence(plt, data, best_checkpoints, accuracy_path, loss_path):
    """Generate separate accuracy and loss convergence plots from epoch data."""
    epochs_range, train_acc, val_acc, train_loss, val_loss = parse_epoch_data(data)

    fig, _ = plot_accuracy(plt, epochs_range, train_acc, val_acc, best_checkpoints)
    fig.savefig(accuracy_path)
    plt.close(fig)

    fig, _ = plot_loss(plt, epochs_range, train_loss, val_loss)
    fig.savefig(loss_path)
    plt.close(fig)


def make_epoch_log_callback(keras, plt, log_path, convergence_dir, model_dir):
    accuracy_path = os.path.join(convergence_dir, "convergence_accuracy.png")
    loss_path = os.path.join(convergence_dir, "convergence_loss.png")

    class EpochLogCallback(keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self._best_val_acc = -1
            self._best_checkpoints = []  # list of (epoch, val_acc)
            self._epoch_start_ts = None
            self._epoch_start_mono = None
            with open(log_path, "w") as f:
                json.dump({}, f)

        def on_epoch_begin(self, epoch, logs=None):
            self._epoch_start_ts = datetime.now().isoformat()
            self._epoch_start_mono = time.monotonic()

        def on_epoch_end(self, epoch, logs=None):
            elapsed = time.monotonic() - self._epoch_start_mono

            # Update epochs.json
            with open(log_path, "r+") as f:
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

            # Save best model by val_acc
            val_acc_cur = logs.get("val_acc", -1)
            if val_acc_cur > self._best_val_acc:
                self._best_val_acc = val_acc_cur
                self._best_checkpoints.append((epoch + 1, val_acc_cur))
                self.model.save(os.path.join(model_dir, f"checkpoint_{epoch + 1}.keras"))

            plot_convergence(plt, data, self._best_checkpoints, accuracy_path, loss_path)

    return EpochLogCallback()
