import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from shared import setup, load_and_visualize, make_model, train, evaluate, make_parser, resolve_args


def run(source_dir, results_dir, epochs, learning_rate, batch_size, validation_split, seed):
    ctx = setup(results_dir)
    train_ds, val_ds, class_names, num_classes, data_aug = load_and_visualize(
        ctx, source_dir, validation_split, seed, batch_size)
    model = make_model(ctx, num_classes)
    train(ctx, model, train_ds, val_ds, epochs, learning_rate, num_classes, results_dir)
    evaluate(ctx, model, val_ds, class_names, num_classes)
    print(f"Results saved to: {results_dir}")


if __name__ == "__main__":
    args = resolve_args(make_parser("Train a Stanford Dogs classifier", "data/stanford_dogs_images"))
    run(args.source_dir, args.output_dir, args.epochs, args.learning_rate,
        args.batch_size, args.validation_split, args.seed)
