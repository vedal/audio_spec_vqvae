import tensorflow as tf
import numpy as np


def binarize(data_dict):
    data_dict['images'] = tf.round(
        tf.cast(data_dict['images'], tf.float32) / tf.constant(255.)
    )
    return data_dict


def input_fn(input_path, batch_size, data_size, cache_dir=False, is_training=True):

    # load cifar as python dicts
    mnist = tf.keras.datasets.mnist
    (train_images, _), (test_images, _) = mnist.load_data()

    train_images, validation_images = np.split(train_images, [40000])

    if is_training:
        input_data_dict = {'images': train_images[..., None]}
    else:
        input_data_dict = {'images': validation_images[..., None]}

    dataset = (
        tf.data.Dataset
        .from_tensor_slices(input_data_dict)
        # 'binarized' MNIST
        .map(binarize)
    )

    if is_training:
        dataset = (
            dataset
            .shuffle(10000)
            .repeat(-1)  # indefinitely
            .batch(batch_size, drop_remainder=False)
            .prefetch(1)
        )
    else:
        dataset = (
            dataset
            .repeat(1)
            .batch(batch_size, drop_remainder=False)
            .prefetch(1)
        )
    return dataset
