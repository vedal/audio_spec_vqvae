# Description
# This script predicts (compress and reconstruct) a new sound clip

import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from input_pipeline_nsynth import input_fn
from utils.utils_sound import save_sound, spectrogram_to_waveform

verbose = False
maxiter_LBFGS = 500

checkpoint_dir = '../results/2019-03-23-sound-latent-size-comparison/trained_further'
#checkpoint_dir = '../results_stored/2019-03-23-sound-latent-size-comparison'
#checkpoint_dir = '../results_stored/2019-02-25-keyboard'

output_dir = 'compress_reconstruct_sonify'

input_path = '../data/test-sounds-for-network/normalized'
p = Path(input_path).glob('*')
dirs = [str(x) for x in p if x.is_dir()]

for sounddir in dirs:

    with tf.Session(graph=tf.Graph()) as session:
        dataset = input_fn(sounddir,
                           is_training=False,
                        data_size=2,
                           batch_size=1)
        next_element = dataset.make_one_shot_iterator().get_next()
        data_norm = session.run(next_element)['images']

        graph_ops = [op.name for op in session.graph.get_operations()]

    specs = {'original': np.squeeze(data_norm)}

    data_norm = np.flipud(data_norm[0, ..., 0])[None, ..., None]

    #meta_models = glob.iglob(checkpoint_dir, recursive=True)
    #for checkpoint_path in meta_models:#[[*meta_models][1]]:#meta_models:
    i = 0
    for checkpoint_path in Path(checkpoint_dir).rglob('*.meta'):
        i += 1
        checkpoint_path = str(checkpoint_path)
        with tf.Session(graph=tf.Graph()) as session:
            saver = tf.train.import_meta_graph(checkpoint_path)
            if not saver:
                continue
            saver.restore(session, checkpoint_path[:-5])

            graph_ops = [op.name for op in session.graph.get_operations()]

            for scope in ['AE', 'VQVAE']:
                try:
                    spec = session.run(
                        scope + '_1/dec/conv2d_transpose_1/BiasAdd:0',
                        {'Placeholder:0': data_norm})

                    spec = np.squeeze(spec)
                    spec = np.flipud(spec)
                    spec = np.clip(spec, 0, 1)
                    specs[scope] = spec

                    # get experiment filename with date,time,hyperparameters
                    experiment_fingerprint = Path(checkpoint_path).parts[-3]

                except Exception as e:
                    if verbose: print(traceback.format_exc())
                    continue

    for specname, spec in specs.items():
        output_path = str(
            Path(output_dir, experiment_fingerprint,
                 Path(sounddir).stem)) + '_{}.wav'.format(specname)

        save_sound(output_path=output_path,
                   waveform=spectrogram_to_waveform(spec,
                                                    apply_l2_norm=True,
                                                    maxiter=maxiter_LBFGS),
                   sample_rate=16000)

    del saver

    if i == 0:
        print('no saved models in', checkpoint_dir)
