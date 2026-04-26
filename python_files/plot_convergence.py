"""Regenerate convergence plots from an existing epochs.json file.

Usage:
    python python_files/plot_convergence.py "data/results/2026-04-08 103742/epochs.json"
    python python_files/plot_convergence.py epochs1.json epochs2.json -o comparison/
"""
import os
import sys
import argparse
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


sys.path.insert(0, os.path.dirname(__file__))
from callbacks import parse_epoch_data, plot_accuracy, plot_loss, plot_convergence


def _reconstruct_best_checkpoints(data):
    """Reconstruct best checkpoints from epoch data."""
    best_checkpoints = []
    best_so_far = -1
    for e in sorted(data.keys(), key=int):
        val_acc = data[e]["vals"]["val_acc"]
        if val_acc > best_so_far:
            best_so_far = val_acc
            best_checkpoints.append((int(e), val_acc))
    return best_checkpoints


def main():
    parser = argparse.ArgumentParser(description="Regenerate convergence plots from epochs.json")
    parser.add_argument("epochs_json", nargs="+",
                        help="Path(s) to epochs.json file(s)")
    parser.add_argument("-o", "--output-dir", default=None,
                        help="Output directory for plots (default: same dir as epochs.json)")
    args = parser.parse_args()

    if len(args.epochs_json) == 1:
        # Single file — regenerate plots in place
        epochs_path = args.epochs_json[0]
        output_dir = args.output_dir or os.path.dirname(epochs_path)
        os.makedirs(output_dir, exist_ok=True)

        with open(epochs_path) as f:
            data = json.load(f)

        plot_convergence(
            plt, data, _reconstruct_best_checkpoints(data),
            accuracy_path=os.path.join(output_dir, "convergence_accuracy.png"),
            loss_path=os.path.join(output_dir, "convergence_loss.png"),
        )
        print(f"Plots saved to: {output_dir}")

    else:
        # Multiple files — regenerate each individually, then overlay comparison
        output_dir = args.output_dir or "."
        os.makedirs(output_dir, exist_ok=True)

        colors = ["green", "blue", "purple", "orange"]
        fig_ax_acc = None
        fig_ax_loss = None

        for idx, epochs_path in enumerate(args.epochs_json):
            with open(epochs_path) as f:
                data = json.load(f)

            label = os.path.basename(os.path.dirname(epochs_path)) or epochs_path
            series = parse_epoch_data(data)
            epochs_range, train_acc, val_acc, train_loss, val_loss = series
            color = colors[idx % len(colors)]

            # Regenerate individual plots (same as training) per run
            run_dir = os.path.join(output_dir, label)
            os.makedirs(run_dir, exist_ok=True)
            plot_convergence(
                plt, data, _reconstruct_best_checkpoints(data),
                accuracy_path=os.path.join(run_dir, "convergence_accuracy.png"),
                loss_path=os.path.join(run_dir, "convergence_loss.png"),
            )

            # Add to comparison plots
            fig_ax_acc = plot_accuracy(plt, epochs_range, train_acc, val_acc,
                                       label=label, train_color=color, val_color=color,
                                       fig_ax=fig_ax_acc)
            fig_ax_loss = plot_loss(plt, epochs_range, train_loss, val_loss,
                                    label=label, train_color=color, val_color=color,
                                    fig_ax=fig_ax_loss)

        fig_ax_acc[0].savefig(os.path.join(output_dir, "comparison_accuracy.png"))
        plt.close(fig_ax_acc[0])

        fig_ax_loss[0].savefig(os.path.join(output_dir, "comparison_loss.png"))
        plt.close(fig_ax_loss[0])

        print(f"Plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
