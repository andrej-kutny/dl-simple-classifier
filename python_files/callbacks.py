import os
import json
import time
from datetime import datetime


def make_epoch_log_callback(keras, plt, log_path, convergence_path, model_dir):
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

            # Update convergence plot from the same data
            epochs_range, train_acc, val_acc, train_loss, val_loss = [], [], [], [], []
            for e, v in data.items():
                epochs_range.append(int(e))
                train_acc.append(v["vals"]["train_acc"])
                val_acc.append(v["vals"]["val_acc"])
                train_loss.append(v["vals"]["train_loss"])
                val_loss.append(v["vals"]["val_loss"])

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            ax1.plot(epochs_range, train_acc, color="lightgreen", label="Train accuracy")
            ax1.plot(epochs_range, val_acc,   color="green",      label="Val accuracy")

            if self._best_checkpoints:
                best_epochs = [p[0] for p in self._best_checkpoints]
                best_accs   = [p[1] for p in self._best_checkpoints]
                ax1.plot(best_epochs, best_accs, color="green", linewidth=0.8,
                         linestyle="--", label="Best checkpoints")
                for ckpt_epoch, ckpt_acc in self._best_checkpoints:
                    ax1.plot(ckpt_epoch, ckpt_acc, "o", color="green", markersize=5)
                    ax1.annotate(f"{ckpt_epoch}: {ckpt_acc:.5f}",
                                 xy=(ckpt_epoch, ckpt_acc),
                                 xytext=(4, 4), textcoords="offset points",
                                 fontsize=7, color="green")

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
            fig.savefig(convergence_path)
            plt.close(fig)

    return EpochLogCallback()
