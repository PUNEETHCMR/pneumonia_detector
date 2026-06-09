import argparse
from pathlib import Path
import tensorflow as tf
from data import get_train_val_datasets, get_test_dataset
from model import build_model, fine_tune_model


def parse_args():
    parser = argparse.ArgumentParser(description='Train pneumonia detection model.')
    parser.add_argument('--data-dir', required=True, help='Root dataset directory containing train/ and val/.')
    parser.add_argument('--test-dir', default=None, help='Optional test dataset root containing test/.')
    parser.add_argument('--output-model', default='model/pneumonia_detector.h5', help='Path to save the trained model.')
    parser.add_argument('--epochs', type=int, default=12, help='Number of training epochs.')
    parser.add_argument('--fine-tune-epochs', type=int, default=5, help='Additional fine-tuning epochs.')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Initial learning rate.')
    return parser.parse_args()


def build_callbacks(output_model_path: Path):
    checkpoint_path = output_model_path.with_suffix('.best.weights.h5')
    return [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(
            str(checkpoint_path),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
    ]


def load_test_dataset(args):
    if args.test_dir:
        return get_test_dataset(args.test_dir)

    candidate = Path(args.data_dir) / 'test'
    if candidate.exists():
        return get_test_dataset(str(candidate))

    return None


def print_metrics(prefix: str, results, metric_names):
    print(f'\n{prefix} metrics:')
    for name, value in zip(metric_names, results):
        print(f'  - {name}: {value:.4f}')


def main():
    args = parse_args()
    output_model_path = Path(args.output_model)
    output_model_path.parent.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds = get_train_val_datasets(args.data_dir)
    test_ds = load_test_dataset(args)

    model = build_model()
    callbacks = build_callbacks(output_model_path)

    print('Starting training...')
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    print('Fine-tuning model...')
    model = fine_tune_model(model, base_trainable=True, learning_rate=1e-5)
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.fine_tune_epochs,
        callbacks=callbacks,
        verbose=1,
    )

    best_weights_path = output_model_path.with_suffix('.best.weights.h5')
    if best_weights_path.exists():
        print(f'Loading best weights from {best_weights_path} before saving final outputs...')
        model.load_weights(str(best_weights_path))

    print('Evaluating validation dataset...')
    val_results = model.evaluate(val_ds, verbose=1)
    print_metrics('Validation', val_results, model.metrics_names)

    if test_ds is not None:
        print('Evaluating test dataset...')
        test_results = model.evaluate(test_ds, verbose=1)
        print_metrics('Test', test_results, model.metrics_names)
    else:
        print('No test dataset found. Skipping test evaluation.')

    try:
        print(f'Saving model weights to: {output_model_path.with_suffix(".weights.h5")}')
        model.save_weights(str(output_model_path.with_suffix('.weights.h5')))

        saved_model_dir = output_model_path.parent / output_model_path.stem
        print(f'Saving full model to: {saved_model_dir}')
        model.save(saved_model_dir, save_format='tf')

        print('✓ Model saved successfully!')
        print(f'  - Weights: {output_model_path.with_suffix(".weights.h5")}')
        print(f'  - SavedModel: {saved_model_dir}')
    except Exception as e:
        print(f'Error saving model: {e}')
        print('Trying to save weights only...')
        try:
            model.save_weights(str(output_model_path))
            print(f'✓ Model weights saved to: {output_model_path}')
        except Exception as e2:
            print(f'Error: {e2}')


if __name__ == '__main__':
    main()
