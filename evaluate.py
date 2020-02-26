# Description
# This script evaluates model performance on a testset

import os
from pathlib import Path
from time import time

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from input_pipeline_nsynth import input_fn
from utils.utils import count_files_in_folder

print('starting script')
last = time()

print('%d - imports' % (time() - last))

# turn off tensorflow logging
tf.logging.set_verbosity(tf.logging.WARN)
# turn off CPU warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

dataset_name = 'nsynth'
image_size, n_channels = (512, 1)
#test_path = '../data/my-keyboard-dataset/test'
test_path = '../data/nsynth-test-vocal/audio'
cache_dir = False # '../cache'

location = 'remote'
if location == 'local':
    results_root = Path('../results_stored/')
    cache_dir = False
    test_size = min(6, count_files_in_folder(test_path))
    batch_size = min(test_size, 2)

elif location == 'remote':
    results_root = Path('../results/')
    test_size = min(100000, count_files_in_folder(test_path))
    batch_size = min(test_size, 16)

else:
    print('wrong location name specified')
    exit(0)

checkpoint_dir = results_root / '2019-02-25-keyboard' #'2019-03-23-sound-latent-size-comparison'
output_dir = Path('script-load-model-run-testset-output')

models_trained_on_upside_down_images = True


def flipfunc(im):
    im['images'] = tf.image.flip_up_down(im['images'])
    return im


# --------------------------------
# train
i = 0
for checkpoint_path in Path(checkpoint_dir).rglob('*.meta'):
    #if ('K=2048' in str(checkpoint_path)) or \
    #   ('D=3' in str(checkpoint_path)): pass
    if ('2019-02-26--T10-14-01' in str(checkpoint_path)):
        pass
    else:
        continue

    i += 1
    testloss = []

    graph = tf.Graph()
    with graph.as_default():
        dataset = input_fn(test_path,
                           is_training=False,
                           cache_dir=cache_dir,
                           data_size=test_size,
                           batch_size=batch_size)
        if models_trained_on_upside_down_images:
            dataset = dataset.map(flipfunc, 4)
        dataset_iterator = dataset.make_initializable_iterator()
        next_element = dataset_iterator.get_next()

        saver = tf.train.import_meta_graph(str(checkpoint_path))
        if not saver:
            print('saver is', saver)
            continue

        x = graph.get_tensor_by_name('Placeholder:0')

        for op in graph.get_operations():
            if '_1/dec/conv2d_transpose_1/BiasAdd' in op.name:
                logits_opname = op.name + ':0'
                break
        logits = graph.get_tensor_by_name(logits_opname)

        sse = tf.reduce_sum((x - logits)**2, axis=(1, 2, 3))

    with tf.Session(graph=graph) as session:
        saver.restore(session, str(checkpoint_path.with_suffix('')))

        dataset_iterator.initializer.run()

        with tqdm(total=test_size) as pbar:
            while 1:
                try:
                    data_norm = session.run(next_element)['images']
                    np.flipud(data_norm[0, ...])

                    error = session.run(sse, {x: data_norm})

                    testloss.append(error)

                    pbar.update(batch_size)

                except tf.errors.OutOfRangeError:
                    break

    testloss = np.concatenate(testloss)

    timestamp_hparams_str = checkpoint_path.parts[-3]
    print(timestamp_hparams_str)
    print('mean_testloss', np.mean(testloss))

    output_path = output_dir / timestamp_hparams_str / dataset_name
