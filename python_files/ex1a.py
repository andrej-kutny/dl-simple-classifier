import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from shared import setup, load_and_visualize, make_model, train, evaluate, make_parser, resolve_args


def run(source_dir, results_dir, epochs, learning_rate, batch_size, validation_split, seed):
    ctx = setup(results_dir)

    # ex1a-specific: filter corrupted images
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
    model = make_model(ctx, num_classes, data_augmentation=data_aug)
    train(ctx, model, train_ds, val_ds, epochs, learning_rate, num_classes, results_dir)
    evaluate(ctx, model, val_ds, class_names, num_classes)
    print(f"Results saved to: {results_dir}")


if __name__ == "__main__":
    args = resolve_args(make_parser("Train a cat/dog classifier", "data/cat_and_dog_images"))
    run(args.source_dir, args.output_dir, args.epochs, args.learning_rate,
        args.batch_size, args.validation_split, args.seed)
