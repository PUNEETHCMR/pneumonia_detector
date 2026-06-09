import pytest
import tensorflow as tf
from src.data import IMAGE_SIZE, BATCH_SIZE, prepare_dataset


def test_prepare_dataset_shape():
    images = tf.random.uniform((2, 256, 256, 3), dtype=tf.float32)
    labels = tf.constant([0, 1], dtype=tf.float32)
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    prepared = prepare_dataset(dataset, augment=False)
    batch = next(iter(prepared))
    assert batch[0].shape[1:] == (*IMAGE_SIZE, 3)
    assert batch[1].shape[0] == 2
