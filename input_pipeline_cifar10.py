import os
import pickle
import numpy as np
import tensorflow as tf
# some functions borrowed from
# https://github.com/deepmind/sonnet/blob/master/sonnet/examples/vqvae_example.ipynb


def unpickle(file_path):
    print("Loading data: " + file_path)

    with open(file_path, mode='rb') as file:
        # In Python 3.x it is important to set the encoding,
        # otherwise an exception is raised here.
        return pickle.load(file, encoding='latin-1')


def reshape_flattened_image_batch(flat_image_batch):
    # convert from NCHW to NHWC
    return flat_image_batch.reshape(-1, 3, 32, 32).transpose([0, 2, 3, 1])


def combine_batches(batch_list):
    images = np.vstack(
        [reshape_flattened_image_batch(batch['data']) for batch in batch_list])
    labels = np.vstack([np.array(batch['labels'])
                        for batch in batch_list]).reshape(-1, 1)
    return {'images': images, 'labels': labels}


def cast_and_normalise_images(data_dict):
    """Convert images to floating point with the range [0.0, 1.0]"""
    images = data_dict['images']
    data_dict['images'] = (tf.cast(images, tf.float32) / 255.0)  # - 0.5
    return data_dict


def input_fn(input_path,batch_size, data_size, cache_dir=False, is_training=True):

    # load cifar batches as python dicts
    n_batches = 1 + int(np.ceil(data_size/10000))

    if is_training:
        # train data dict
        input_data_dict = combine_batches([
            unpickle(os.path.join(input_path,
                                  'cifar-10-batches-py/data_batch_%d' % i)) for i in range(1, n_batches)])
    else:
        input_data_dict = combine_batches([
            unpickle(os.path.join(input_path,
                                  'cifar-10-batches-py/data_batch_5'))])

    dataset = (
        tf.data.Dataset
        .from_tensor_slices(input_data_dict)
        .map(cast_and_normalise_images)
    )

    if is_training:
        dataset = (
            dataset
            .shuffle(1000)
            .repeat(-1)  # indefinitely
            .batch(batch_size, drop_remainder=False)
            .prefetch(1))
    else:
        dataset = (
            dataset
            .repeat(1)
            .batch(batch_size, drop_remainder=False)
            .prefetch(1)
        )
    return dataset


def test_pipeline(local_data_dir, download_data=False):
    if local_data_dir == '':
        local_data_dir = '../data/cifar10/'

    if download_data:
        # Download CIFAR10
        data_path = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

        local_data_dir = tempfile.mkdtemp()  # Change this as needed
        tf.gfile.MakeDirs(local_data_dir)

        url = urllib.urlopen(data_path)
        archive = tarfile.open(fileobj=url,
                               mode='r|gz')  # read a .tar.gz stream
        archive.extractall(local_data_dir)
        url.close()
        archive.close()
        print('extracted data files to %s' % local_data_dir)

    # dataset iterators
    dataset_train = input_fn(local_data_dir, batch_size=128, cache_dir=False)
    dataset_train_iterator = dataset_train.make_one_shot_iterator()
    dataset_train_batch = dataset_train_iterator.get_next()

    with tf.Session() as sess:
        batch = sess.run(dataset_train_batch)

        import matplotlib.pyplot as plt
        I = batch['images'][50]
        plt.imshow(I)
