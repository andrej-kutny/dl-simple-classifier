import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from shared import (
    setup, load_and_visualize, load_and_adapt_model,
    train, evaluate, make_transfer_parser, resolve_args,
)


def run(model_path, source_dir, results_dir, epochs, learning_rate, batch_size, validation_split, seed):
    ctx = setup(results_dir)

    # Filter corrupted images
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

    train_ds, val_ds, class_names, num_classes, data_aug = load_and_visualize(
        ctx, source_dir, validation_split, seed, batch_size)

    model = load_and_adapt_model(ctx, model_path, num_classes, reinit_conv=-2)

    train(ctx, model, train_ds, val_ds, epochs, learning_rate, num_classes, results_dir)
    evaluate(ctx, model, val_ds, class_names, num_classes)
    print(f"Results saved to: {results_dir}")


if __name__ == "__main__":
    args = resolve_args(make_transfer_parser("Ex4: Transfer learning — replace output + last two conv layers"))
    run(args.model, args.source_dir, args.output_dir, args.epochs, args.learning_rate,
        args.batch_size, args.validation_split, args.seed)
