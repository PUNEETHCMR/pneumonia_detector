import os
from typing import Tuple
import tensorflow as tf


IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# Handle AUTOTUNE compatibility across TensorFlow versions
try:
    AUTOTUNE = tf.data.AUTOTUNE
except AttributeError:
    try:
        AUTOTUNE = tf.data.experimental.AUTOTUNE
    except AttributeError:
        AUTOTUNE = -1  # Fallback for older versions


def build_image_dataset(data_dir: str, subset: str = '', seed: int = 42) -> tf.data.Dataset:
    """Build a tf.data pipeline for image classification."""
    path = os.path.join(data_dir, subset) if subset else data_dir
    try:
        # Try newer TensorFlow API first
        return tf.keras.utils.image_dataset_from_directory(
            path,
            labels='inferred',
            label_mode='binary',
            image_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            shuffle=True,
            seed=seed,
        )
    except AttributeError:
        # Fallback for older versions
        return tf.keras.preprocessing.image_dataset_from_directory(
            path,
            labels='inferred',
            label_mode='binary',
            image_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            shuffle=True,
            seed=seed,
        )


def prepare_dataset(dataset: tf.data.Dataset, augment: bool = False) -> tf.data.Dataset:
    """Prepare dataset with resizing, rescaling, and optional augmentation."""
    def _process(image, label):
        image = tf.image.resize(image, IMAGE_SIZE)
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    dataset = dataset.map(_process, num_parallel_calls=AUTOTUNE)
    if augment:
        dataset = dataset.map(_augment, num_parallel_calls=AUTOTUNE)
    return dataset.cache().prefetch(AUTOTUNE)


def _augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.2, 0.9)
    return image, label


def get_train_val_datasets(data_dir: str) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    train_ds = build_image_dataset(data_dir, 'train')
    val_ds = build_image_dataset(data_dir, 'val')
    return prepare_dataset(train_ds, augment=True), prepare_dataset(val_ds, augment=False)


def get_test_dataset(data_dir: str) -> tf.data.Dataset:
    test_ds = build_image_dataset(data_dir)
    return prepare_dataset(test_ds, augment=False)
