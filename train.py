import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path
from time import time

import numpy as np
import tensorflow as tf
import yaml
from tqdm import trange

from models import loss_fn, model_fn
from utils.utils_logging import (get_current_filename,
                                 get_date_time_hparams_as_str, init_logging,
                                 print_and_log, save_experiment_src_as_zip)


if __name__ == '__main__':
    print('starting script')
    last = time()

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-f", "--file",
                        dest="yaml_path",
                        help="experiment definition file yaml",
                        metavar="FILE",
                        required=True)
    args = parser.parse_args()

    # ---------------- EXPERIMENT SETUP ----------------

    # load config yaml
    with open(args.yaml_path, 'r') as stream:
        cfg = yaml.safe_load(stream)

    checkpoint_dir = None

    save_plots = cfg['save_plots']

    if save_plots:
        from utils.utils_plotting import (plot_loss_with_annotated_min,
                                          plot_perplexity, close_plots)

    if cfg['dataset'] == 'mnist':
        from input_pipeline_mnist import input_fn
    elif cfg['dataset'] == 'cifar10':
        from input_pipeline_cifar10 import input_fn
    elif cfg['dataset'] == 'nsynth':
        from input_pipeline_nsynth import input_fn
    else:
        print('wrong dataset name specified')
        exit(0)

    print('%d - imports' % (time() - last))

    for model_type, K, D in [(m, K, D) for m in ['VQVAE', 'AE']
                             for K in cfg[m]['K'] for D in cfg[m]['D']]:

        print(model_type, K, D)
        hp = {  # hyperparameters
            'model_type': model_type,
            'K': K,
            'D': D,
            'lr': cfg['learning_rate'],
            'beta': cfg['beta'],
            'batch_size': cfg['batch_size'],
            'n_units': cfg['n_units'],
        }

        # -------------------------------
        # Logging
        if save_plots:
            logdirname = Path(get_current_filename(),
                              get_date_time_hparams_as_str(hp))

            # plots, checkpoints and .py files used
            results_dir = Path(cfg['results_root']) / logdirname
            results_dir.mkdir(parents=True, exist_ok=True)

            # save zip dir containing experiment files
            save_experiment_src_as_zip(results_dir)

            # tensorboard summaries
            logdir_tb = Path(cfg['results_root']) / 'tensorboard' / logdirname
            logdir_tb.mkdir(parents=True, exist_ok=True)

            print(f'{time() - last} - logging results to', results_dir)
            print(f'{time() - last} - logging tensorboard to', logdir_tb)

        # -----------------------------
        # create graph

        graph = tf.Graph()
        with graph.as_default():
            global_step = tf.train.get_or_create_global_step()

            # Data input pipeline
            with tf.device('/cpu:0'):
                with tf.name_scope(None, default_name='training_data'):
                    # dataset iterators
                    dataset_train = input_fn(cfg['train_path'], cfg['batch_size'],
                                             cfg['train_size'], cfg['cache_dir'])
                    dataset_train_iterator = dataset_train.make_one_shot_iterator()
                    dataset_train_batch = dataset_train_iterator.get_next()

                with tf.name_scope(None, default_name='validation_data'):
                    dataset_test = input_fn(cfg['test_path'], cfg['batch_size'],
                                            cfg['test_size'], cfg['cache_dir'],
                                            is_training=False)
                    dataset_test_iterator = dataset_test.make_initializable_iterator()
                    dataset_test_batch = dataset_test_iterator.get_next()

            def get_images(sess, subset='train'):
                if subset == 'train':
                    return sess.run(dataset_train_batch)['images']
                elif subset == 'valid':
                    return sess.run(dataset_test_batch)['images']

            # data input
            img_shape = (cfg['image_size'], cfg['image_size'], cfg['n_channels'])

            x = tf.placeholder(tf.float32, shape=(None, *img_shape))

            # train network
            with tf.variable_scope(model_type):
                logits, bottleneck_output = model_fn(x, hp, is_training=True)
            # test network
            with tf.variable_scope(model_type, reuse=True):
                logits_test, bottleneck_output_test = model_fn(x, hp, is_training=False)

            with tf.variable_scope('Optimization'):
                ae_loss = loss_fn(x, logits, bottleneck_output, hp)

                optimizer = tf.train.AdamOptimizer(hp['lr'])
                train_op = optimizer.minimize(ae_loss, global_step=global_step, name='train_op')

            with tf.variable_scope('Metrics'):
                sse = tf.reduce_sum((x - logits_test)**2, axis=[1, 2, 3])
                mse = tf.reduce_mean(sse)
                tf.summary.scalar('mse', mse)

                if model_type == 'VQVAE':
                    # get the frequency of each latent vector during training
                    tf.summary.scalar('perplexity_train',
                                      bottleneck_output["perplexity"])
                    tf.summary.scalar('perplexity_test',
                                      bottleneck_output_test["perplexity"])

                # add image reconstruction to tensorboard
                image_x = tf.cast(x * 255., tf.uint8)
                image_logits_test = tf.cast(logits_test * 255., tf.uint8)
                tf.summary.image('original', image_x, max_outputs=1)
                tf.summary.image('reconstruction',
                                 image_logits_test,
                                 max_outputs=1)

                merged = tf.summary.merge_all()
                if save_plots:
                    train_writer = tf.summary.FileWriter(str(logdir_tb) + '/',
                                                         graph=None)

            saver = tf.train.Saver(max_to_keep=2)

        print('%d - finished model definition' % (time() - last))

        # --------------------------------
        # train
        with tf.Session(graph=graph) as session:
            tf.local_variables_initializer().run()
            tf.global_variables_initializer().run()

            # restore checkpoint
            if checkpoint_dir:
                checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir +
                                                             'save_models/')
                saver.restore(session, checkpoint_path)
                print('loaded saved model ', checkpoint_path)

            losses = []
            train_perpl = []
            start_time = time()

            # Print log headers
            fmt = init_logging(cfg['logpath'])

            try:
                for epoch in range(cfg['epochs']):
                    for _ in trange(cfg['train_size'] // hp['batch_size']):

                        fd = {x: get_images(session, subset='train')}

                        if model_type == 'VQVAE':
                            _, pp = session.run(
                                [train_op, bottleneck_output['perplexity']], fd)
                            train_perpl.append(pp)

                        else:
                            session.run(train_op, fd)

                    # run test set
                    dataset_test_iterator.initializer.run()
                    print('validation...')
                    test_loss = []
                    while 1:
                        try:
                            fd = {x: get_images(session, subset='valid')}
                            test_loss.append(session.run(mse, fd))
                        except tf.errors.OutOfRangeError:
                            break

                    if save_plots:
                        summary, step = session.run([merged, global_step], fd)
                        train_writer.add_summary(summary, step)

                    loss = np.mean(test_loss)
                    losses.append(loss)

                    if min(losses[:-1], default=1) > loss:

                        # save model checkpoint
                        if cfg['save_checkpoints']:
                            print(
                                saver.save(session,
                                           results_dir + 'save_models/model.ckpt',
                                           global_step=epoch))

                        # plot matplotlib figures
                        if save_plots:
                            plot_loss_with_annotated_min(losses,
                                                         save_path=results_dir)
                            if model_type == 'VQVAE':
                                plot_perplexity(train_perpl, results_dir)

                            close_plots()

                    # print epoch report
                    print_and_log(
                        fmt.format(elapsed_time=(time() - start_time) / 60,
                                   epoch=epoch,
                                   loss=loss,
                                   min_loss_epoch=np.argmin(losses)))

            except KeyboardInterrupt:
                pass
            finally:
                print_and_log('finished')
