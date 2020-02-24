import tensorflow as tf
import sonnet as snt
from functools import partial

conv2d = partial(
    tf.layers.conv2d,
    padding="SAME",
    kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))

conv2d_transpose = partial(
    tf.layers.conv2d_transpose,
    padding="SAME",
    kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))


def _res_block(x_in, n_units):
    x = tf.nn.relu(x_in)
    x = conv2d(x, filters=n_units, kernel_size=(3, 3), strides=(1, 1), activation=None)
    x = tf.nn.relu(x)
    x = conv2d(x, filters=n_units, kernel_size=(1, 1), strides=(1, 1), activation=None)
    return x + x_in


def _encoder(x, n_units):
    with tf.variable_scope('enc'):
        x = conv2d(x, filters=n_units, kernel_size=(4, 4), strides=(2, 2), activation=tf.nn.relu)
        x = conv2d(x, filters=n_units, kernel_size=(4, 4), strides=(2, 2), activation=tf.nn.relu)
        x = _res_block(x, n_units)
        x = _res_block(x, n_units)
    return x


def _decoder(x, n_units, n_channels):
    with tf.variable_scope('dec'):
        x = conv2d(x, filters=n_units, kernel_size=(3, 3), strides=(1, 1), activation=None)
        x = _res_block(x, n_units)
        x = _res_block(x, n_units)
        x = conv2d_transpose(x, filters=n_units, kernel_size=(4, 4), strides=(2, 2), activation=tf.nn.relu)
        x = conv2d_transpose(x, filters=n_channels, kernel_size=(4, 4), strides=(2, 2), activation=None)
    return x


def model_fn(x, hp, is_training=True):
    n_channels = x.get_shape()[-1]
    x = _encoder(x, hp['n_units'])
    z_e = conv2d(x, filters=hp['D'], kernel_size=(1, 1), strides=(1, 1), activation=None)

    bottleneck_output = {}
    if hp['model_type'] == 'VQVAE':
        vq_vae = snt.nets.VectorQuantizerEMA(
            embedding_dim=hp['D'],
            num_embeddings=hp['K'],
            commitment_cost=hp['beta'],
            decay=0.99)
        vq_output = vq_vae(z_e, is_training=is_training)
        bottleneck_output = vq_output

        z = vq_output["quantize"]

    elif hp['model_type'] == 'VAE':
        z_mu = tf.identity(z_e, name='z_mu')
        z_logvar = conv2d(x, filters=hp['D'], kernel_size=(1, 1), strides=(1, 1), activation=None, name='z_logvar')
        bottleneck_output = {'z_mu': z_mu, 'z_logvar': z_logvar}

        # Reparametrization
        eps = tf.random_normal(shape=tf.shape(z_mu), name='eps')
        z_std = tf.exp(0.5*z_logvar)  # div 2 comes from log(var^2)
        z = tf.add(z_mu, z_std * eps, name='z')

    else:  # AE
        z = z_e

    print('z.shape', z.shape)

    # Decoder
    logits = _decoder(z, hp['n_units'], n_channels)

    return logits, bottleneck_output


def loss_fn(x, logits, bottleneck_output, hp):
    if hp['model_type'] == 'VQVAE':
        reconstruction_error = tf.reduce_mean((logits - x)**2)
        vq_loss = bottleneck_output['loss']
        loss = reconstruction_error + vq_loss

        tf.summary.scalar('vq_loss', vq_loss)

    else:  # AE | VAE
        reconstruction_error = tf.reduce_sum((logits - x)**2, axis=[1, 2, 3])
        KLD = 0
        if hp['model_type'] == 'VAE':
            z_mu = bottleneck_output['z_mu']
            z_logvar = bottleneck_output['z_logvar']
            # KL divergence
            # Kingma and Welling. Auto-Encoding Variational Bayes (Appendix B)
            KLD = -0.5 * tf.reduce_sum(
                1 + z_logvar - z_mu**2 - tf.exp(z_logvar),
                axis=[1, 2, 3],
                name='KLD')

            tf.summary.scalar('KLD', KLD)

        loss = tf.reduce_mean(reconstruction_error + KLD)

    return loss
