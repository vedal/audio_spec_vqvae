import traceback
from pathlib import Path
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from input_pipeline_nsynth import input_fn

from utils.utils_sound import save_sound, spectrogram_to_waveform

if __name__ == '__main__':
    parser = ArgumentParser(
        description="predicts (compress and reconstruct) a new sound clip using a pretrained model",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint_dir",
        dest="checkpoint_dir",
        default="../results/2019-03-23-sound-latent-size-comparison/trained_further",
        #default="../results/2019-03-23-sound-latent-size-comparison",
        #default="../results/2019-02-25-keyboard",
        help="model checkpoint directory",
        required=False,
        type=Path,
    )

    parser.add_argument(
        "--maxiter",
        dest="maxiter",
        help="maximum L-BFGS-B iterations to reconstruct the audio sample",
        default=500,
        required=False,
        type=int,
    )

    parser.add_argument(
        "--data_dir",
        dest="data_dir",
        default="../data/test-sounds-for-network/normalized",
        help="input data directory",
        required=False,
        type=Path,
    )

    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        default="compress_reconstruct_sonify",
        help="output directory",
        type=Path,
        required=False,
    )

    parser.add_argument("--verbose", help="increase output verbosity", action="store_true")

    args = parser.parse_args()

    verbose = args.verbose
    maxiter_LBFGS = args.maxiter

    dirs = [x for x in args.data_dir.glob("*") if x.is_dir()]

    for sounddir in dirs:

        with tf.Session(graph=tf.Graph()) as session:
            dataset = input_fn(str(sounddir), is_training=False, data_size=2, batch_size=1)
            next_element = dataset.make_one_shot_iterator().get_next()
            data_norm = session.run(next_element)["images"]

            graph_ops = [op.name for op in session.graph.get_operations()]

        specs = {"original": np.squeeze(data_norm)}

        data_norm = np.flipud(data_norm[0, ..., 0])[None, ..., None]

        i = 0
        for checkpoint_path in args.checkpoint_dir.rglob("*.meta"):
            i += 1
            checkpoint_path = str(checkpoint_path)
            with tf.Session(graph=tf.Graph()) as session:
                saver = tf.train.import_meta_graph(checkpoint_path)
                if not saver:
                    continue
                saver.restore(session, checkpoint_path[:-5])

                graph_ops = [op.name for op in session.graph.get_operations()]

                for scope in ["AE", "VQVAE"]:
                    try:
                        spec = session.run(
                            scope + "_1/dec/conv2d_transpose_1/BiasAdd:0",
                            {"Placeholder:0": data_norm},
                        )

                        spec = np.squeeze(spec)
                        spec = np.flipud(spec)
                        spec = np.clip(spec, 0, 1)
                        specs[scope] = spec

                        # get experiment filename with date,time,hyperparameters
                        experiment_fingerprint = Path(checkpoint_path).parts[-3]

                    except Exception as e:
                        if verbose:
                            print(traceback.format_exc())
                        continue

        for specname, spec in specs.items():
            output_path = str(
                Path(args.output_dir, experiment_fingerprint, sounddir.stem)
            ) + "_{}.wav".format(specname)

            save_sound(
                output_path=output_path,
                waveform=spectrogram_to_waveform(
                    spectrogram=spec, apply_l2_norm=True, maxiter=maxiter_LBFGS
                ),
                sample_rate=16000,
            )

        del saver

        if i == 0:
            print("no saved models in", args.checkpoint_dir)
