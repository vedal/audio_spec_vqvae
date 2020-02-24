import numpy as np
import tensorflow as tf
from librosa.display import specshow


def log10(x):
    return tf.log(x) / tf.log(10.0)


def rms(waveform):
    return tf.sqrt(tf.reduce_mean(waveform**2))


def load(audio_file_path, sample_rate=16000):
    """Load wav files as single-channel (mono) waveform (vector)"""
    x = tf.read_file(audio_file_path)
    x = tf.contrib.ffmpeg.decode_audio(x, 'wav', sample_rate, channel_count=1)
    x = tf.transpose(x)
    return x


def segment(waveform, duration=4.152, sample_rate=16000):
    """Convert waveform into list of segments
    
    Args:
        waveform (tf.Tensor float32): Complete audio waveform of one sound file
        duration (float, optional): Length of each segment in seconds. Defaults to 4.152.
        sample_rate (int, optional): Sampling rate of audio waveform. Defaults to 16000.
    
    Returns:
        DatasetV1Adapter('float32'): dataset of waveform segments
    """

    num_samples = int(sample_rate * duration)
    x = tf.signal.frame(waveform, num_samples, num_samples, pad_end=True)
    frames = tf.shape(x, out_type=tf.int64)[1]
    dataset = (
        tf.data.Dataset
        .range(frames)
        .map(lambda i: tf.gather(x, i, axis=1))
        .filter(lambda x: rms(x) > 0.01) # filter out silence
    )

    return dataset


def transform(waveform):
    """Transform waveform segments into real-valued spectrograms (lossy)"""
    z = tf.signal.stft(waveform,
                       frame_length=1024,
                       frame_step=128,
                       fft_length=1024)
    magnitudes = tf.abs(z)
    power = magnitudes**2

    # convert to decibels (logarithmic)
    # same as librosa.core.power_to_db
    top_db = 120.0
    amin = 1e-9
    spectrogram = 10.0 * log10(tf.maximum(power, amin))
    spectrogram /= top_db

    return spectrogram


def minmax(spectrogram):
    x = spectrogram
    x -= tf.reduce_min(x)
    x /= tf.reduce_max(x)
    return x


def transpose_and_crop(z):
    z = tf.transpose(z, perm=[2, 1, 0])
    z = z[:-1]  # crop out highest frequency component
    return z


def to_dict(spectrogram):
    return {'images': spectrogram}


def input_fn(input_path, batch_size, data_size, cache_dir=None, is_training=True):
    """
    Convert *.wav files of the given folder into 512x512 spectrograms.

    Input data can be of any length. The pipeline splits it into chunks of 4.152 seconds. 

    Big thanks to Carl Thome and Agril Hilmkil for helping to make this pipeline
    
    Args:
        input_path (str): folder path to sound files. No trailing slash
        batch_size (int):
        data_size (int): total number of datapoints in the dataset
        is_training (bool, optional): Defaults to True.
        cache_dir ([type], optional): Include a directory for cached dataset. Defaults to None.
    
    Returns:
        dict: {'images': spectrograms}

    """

    if data_size < 2:
        # avoid OutOfRange for .
        data_size = 2

    interleave = tf.contrib.data.parallel_interleave
    num_parallel_calls = 4
    dataset = (
        tf.data.Dataset
        .list_files(input_path + '/*')
        .take(data_size)
        .shuffle(min(data_size, 100))
        .map(load, num_parallel_calls)
        .apply(interleave(segment, batch_size))
        .map(transform, num_parallel_calls)
        .map(minmax, num_parallel_calls)
        .map(transpose_and_crop)
        .map(to_dict, num_parallel_calls)
    )

    # for small datasets without cacheing (mainly for testing)
    if cache_dir is None:
        if is_training:
            dataset = dataset.repeat(-1)
        return dataset.batch(batch_size=batch_size, drop_remainder=True)

    if is_training:
        print('is_training')
        dataset = (
            dataset
            .cache(cache_dir + '/training')
            .cache() # cache in memory
            .shuffle(min(data_size, 64))
            .repeat(-1)  # indefinitely
            .batch(batch_size, drop_remainder=True)
            .prefetch(1)
        )
    else:
        dataset = (
            dataset
            .cache(cache_dir + '/validation')
            .batch(batch_size, drop_remainder=False)
            .prefetch(1)
            )

    return dataset


def test_pipeline(path='../data/my-keyboard-dataset/train'):
    import matplotlib.pyplot as plt
    graph = tf.Graph()
    with graph.as_default():
        dataset = input_fn(path, batch_size=1, data_size=2, cache_dir=None)
        element = dataset.make_one_shot_iterator().get_next()

    with tf.Session(graph=graph):
        spectrograms = element['images'].eval()
        print(spectrograms.shape, spectrograms.min(), spectrograms.max())
        print('spectrogram_var', np.var(spectrograms[0, ..., 0]))
        assert spectrograms.min() >= 0.0
        assert spectrograms.max() <= 1.0
        specshow(spectrograms[0, ..., 0])
        plt.show()


if __name__ == '__main__':
    test_pipeline()
