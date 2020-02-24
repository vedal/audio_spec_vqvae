import tensorflow as tf
import numpy as np


def negLL_latents_softmaxCE(emp_distr, latent_encodings_oh, batch_size, K):
    """ estimates negative log-likelihood of latent encoding in VQVAE, as explained by Oord in emails:
	
	Oord email:
		For the latents we also put an independent categorical distribution (so softmax again) 
        over the latents (so just counting how many times every latent occurs).

	Oord email 2:
		The counts come from the training set. Let's say the count probs are 
		p_1, p_2, ..., p_K, for K possible discrete values.
		
		Then the loss is simply Sum_j -log(p_ij), where i is the cluster chosen for 
        datapoint j (doesn't matter if it's test or train).
		
		Then you do this for every latent in the model 
	
	Arguments:
		emp_distr {np.array} -- empirical distribution over latent vectors from train set. shape=(K,)
		latent_encodings_oh {np.array} -- one-hot encoded latent vector encodings for test set batch. 
            shape=(batch_size * z_height * z_width, K)
        batch_size {int}
        K {int} -- number of latent vectors in cookbook
	
	Returns:
		[float] -- negative log-likelihood
	"""

    # Divide the latent encodings into batches, where each batch contains all
    # latent encodings for image x_i
    # This will enable taking the mean over the batch dimension after summing over others.
    latent_encodings_oh = tf.reshape(latent_encodings_oh,
                                     shape=(batch_size, -1, K))

    # note: no need to stack the emp_distr to match latent encodings.
    # it seems to broadcast correctly.
    return tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(
        labels=emp_distr, logits=latent_encodings_oh),
        axis=[1])


def negLL_latents_softmaxCE_2(latent_counts, latent_encodings_oh, batch_size,
                              K):
    emp_distr = tf.nn.softmax(latent_counts) + 1.e-10

    latent_encodings = tf.argmax(latent_encodings_oh, axis=1)

    h = tf.nn.embedding_lookup(emp_distr, latent_encodings)

    i = tf.reshape(h, shape=(batch_size, -1))

    # cross entropy
    return tf.reduce_sum(-tf.log(i), axis=[1])


def negLL_latents_2(emp_distr, latent_encodings_oh, batch_size, K):
    # Divide the latent encodings into batches, where each batch contains all
    # latent encodings for image x_i
    # This will enable taking the mean over the batch dimension after summing over others.
    latent_encodings_oh = tf.reshape(latent_encodings_oh,
                                     shape=(batch_size, -1, K))

    nll = -tf.reduce_sum(emp_distr * latent_encodings_oh, axis=[1, 2])

    return nll


def negLL_latents_3(emp_logits, latent_encodings_oh, batch_size, K):
    # Divide the latent encodings into batches, where each batch contains all
    # latent encodings for image x_i
    # This will enable taking the mean over the batch dimension after summing over others.
    latent_encodings_oh = tf.reshape(latent_encodings_oh,
                                     shape=(batch_size, -1, K))

    emp_distr = tf.nn.softmax(emp_logits)

    # note: no need to stack the emp_distr to match latent encodings.
    # it seems to broadcast correctly.
    return tf.reduce_sum(emp_distr * latent_encodings_oh, axis=[1, 2])


def ll_nats_to_nll_bits_dim(ll):
    """Convert log-likelihood in nats to negative log-likelihood in bits/dim for CIFAR-10
	
	Arguments:
		ll {float} -- log-likelihood (nats).
	
	Returns:
		[float] -- negative log-likelihood, rescaled (bits/dim).

	Reference:
	psamba: www.reddit.com/r/MachineLearning/comments/56m5o2
	"""

    # -((logL_base_e) / num_dim) - log_e(scaling) ) / log_e(2)
    return -(ll / (3 * 32 * 32) - np.log(128.)) / np.log(2)


def ll_nats_to_nll_bits_dim_inv(nll):
    return -(nll * np.log(2.) - np.log(128.)) * (3 * 32 * 32)


def make_metrics_dict(metrics):
    return dict(zip([v.name.split('/')[2] for v in metrics], metrics))


def summarize_metrics(metrics_dict):
    for name, metric in metrics_dict.items():
        tf.summary.scalar(name, metric)


def calc_dataset_variance_2(input_fn, data_path, data_size, batch_size,
                            cache_dir):
    graph = tf.Graph()
    with graph.as_default():
        dataset = input_fn(data_path, batch_size, cache_dir=cache_dir)
        dataset_batch = dataset.make_one_shot_iterator().get_next()

    # calculate variance of training dataset
    def mysum_matrix(l, s=0, s2=0, N=0):
        for e in l:
            s += e
            s2 += e * e
            N += 1
        return (s, s2, N)

    s, s2, N = 0, 0, 0
    print('calculating variance...')

    with tf.Session(graph=graph) as sess:
        for _ in trange(data_size // batch_size):
            d = sess.run(dataset_batch)
            s, s2, N = mysum_matrix(d.reshape(-1), s, s2, N)

    var = lambda s, s2, N: (s2 - (s * s) / N) / N

    data_variance = var(s, s2, N)

    print('done')
    print('s={}, s2={}, N={}'.format(s, s2, N))
    print('dataset variance', data_variance)

    return data_variance
