import os
from pathlib import Path
from time import time
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from input_pipeline_nsynth import input_fn
from utils.utils import count_files_in_folder


def flipfunc(im):
    im["images"] = tf.image.flip_up_down(im["images"])
    return im


if __name__ == "__main__":

    # disable tensorflow logging
    tf.logging.set_verbosity(tf.logging.WARN)
    # disable CPU warning
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    parser = ArgumentParser(
        description="evaluate pretrained model performance on a testset",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data_dir",
        dest="data_dir",
        help="dir of test audio files",
        default="../data/my-keyboard-dataset/test",
        type=Path,
    )

    parser.add_argument(
        "--checkpoint_dir",
        dest="checkpoint_dir",
        help="model checkpoint directory",
        default="../results/2019-02-25-keyboard",
        #'2019-03-23-sound-latent-size-comparison'
        type=Path,
    )

    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        help="directory for output files",
        default="script-load-model-run-testset-output",
        type=Path,
    )

    parser.add_argument("--cache_dir", dest="cache_dir", help="Flag to use cache")

    parser.add_argument(
        "--debug",
        dest="debug",
        help="Flag to use small part of dataset",
        default=False,
        action="store_true",
    )

    args = parser.parse_args()
    print(args.data_dir)
    max_testset_size, max_batch_size = (6, 2) if args.debug else (100000, 16)
    test_size = min(max_testset_size, count_files_in_folder(args.data_dir))
    batch_size = min(max_batch_size, test_size)

    models_trained_on_upside_down_images = True

    print("starting script")
    last = time()

    # --------------------------------
    # train
    i = 0
    for checkpoint_path in args.checkpoint_dir.rglob("*.meta"):

        i += 1
        testloss = []

        graph = tf.Graph()
        with graph.as_default():
            dataset = input_fn(
                str(args.data_dir),
                is_training=False,
                cache_dir=args.cache_dir,
                data_size=test_size,
                batch_size=batch_size,
            )
            if models_trained_on_upside_down_images:
                dataset = dataset.map(flipfunc, 4)
            dataset_iterator = dataset.make_initializable_iterator()
            next_element = dataset_iterator.get_next()

            saver = tf.train.import_meta_graph(str(checkpoint_path))
            if not saver:
                print("saver is", saver)
                continue

            x = graph.get_tensor_by_name("Placeholder:0")

            for op in graph.get_operations():
                if "_1/dec/conv2d_transpose_1/BiasAdd" in op.name:
                    logits_opname = op.name + ":0"
                    break

            logits = graph.get_tensor_by_name(logits_opname)

            sse = tf.reduce_sum((x - logits) ** 2, axis=(1, 2, 3))

        with tf.Session(graph=graph) as session:
            saver.restore(session, str(checkpoint_path.with_suffix("")))

            dataset_iterator.initializer.run()

            with tqdm(total=test_size) as pbar:
                while 1:
                    try:
                        data_norm = session.run(next_element)["images"]
                        np.flipud(data_norm[0, ...])

                        error = session.run(sse, {x: data_norm})

                        testloss.append(error)

                        pbar.update(batch_size)

                    except tf.errors.OutOfRangeError:
                        break

        testloss = np.concatenate(testloss)

        timestamp_hparams_str = checkpoint_path.parts[-3]
        print(timestamp_hparams_str)
        print("mean_testloss", np.mean(testloss))
