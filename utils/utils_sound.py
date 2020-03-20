from pathlib import Path
import tensorflow as tf


def sonify(spectrogram,
           samples,
           transform_op_fn,
           apply_l2_norm=True,
           maxiter=1000,
           logscaled=True):
    
    # Big thanks to Carl Thome for help with this function
        
    graph = tf.Graph()
    with graph.as_default():

        noise = tf.Variable(tf.random_normal([samples], stddev=1e-6))

        x = transform_op_fn(noise, apply_l2_norm)
        y = spectrogram

        # log-scaled values should be scaled back.
        # (log scaling happens in the transform function in pipeline as well)
        # this part should only be active if there's log scaling in the pipeline transform!
        #if logscaled:
        #    x = tf.expm1(x)
        #    y = tf.expm1(y)
        x = tf.sqrt(10**((120 * x) / 10))
        y = tf.sqrt(10**((120 * y) / 10))

        # normalizing here instead of in spectrogram_to_waveform produces a more crisp-
        # sounding output.

        if apply_l2_norm:
            x = tf.nn.l2_normalize(x)
            y = tf.nn.l2_normalize(y)

        tf.losses.mean_squared_error(x, y)

        optimizer = tf.contrib.opt.ScipyOptimizerInterface(
            loss=tf.losses.get_total_loss(),
            var_list=[noise],
            tol=1e-16,
            #tol=1e-16,
            method='L-BFGS-B',
            options={
                'maxiter': maxiter,
                'disp': True,
            })

    with tf.Session(graph=graph) as session:
        session.run(tf.global_variables_initializer())
        optimizer.minimize(session)
        waveform = session.run(noise)

    return waveform


def sonify_transf_2(waveform, apply_l2_norm=True):
    from input_pipeline_nsynth import minmax, transform, transpose_and_crop
    spectrogram = transform(waveform)
    if not apply_l2_norm:
        spectrogram = minmax(spectrogram)  #[:,:-1] # same as channel_dim_last
    spectrogram = spectrogram[None, ...]
    spectrogram = transpose_and_crop(spectrogram)

    return spectrogram


def spectrogram_to_waveform(spectrogram, apply_l2_norm=True, maxiter=1000):
    # input reconstructed spectrogram and output a waveform
    from utils.utils import minmax
    if not apply_l2_norm:
        x = minmax(spectrogram)[..., None]
    else:
        x = spectrogram[..., None]

    return sonify(x, 66432, sonify_transf_2, apply_l2_norm, maxiter)


def save_sound(output_path, waveform, sample_rate, normalize=True):
    # save waveform to .wav sound file
    # example:
    #   save_sound('../output',output.wav,waveform,sample_rate)

    # ensure that output_dir exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    from librosa.output import write_wav
    write_wav(output_path, waveform, sr=sample_rate, norm=normalize)

    print(output_path, 'saved')

    return
