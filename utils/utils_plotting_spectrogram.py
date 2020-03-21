import numpy as np
import matplotlib.pyplot as plt


def spec_plotter(ax, data, param_dict={}):
    "Plot single spectrogram with librosa specshow, specifying title"

    import librosa
    from librosa.display import specshow

    try:
        ax.set_title(param_dict["title"])
    except:
        pass
    ax = specshow(data, ax=ax)
    return ax


def freq_response_plotter(ax, data1, data2, param_dict={}):
    # Plot vertical frequency response plot of MSE between two spectrograms.
    # Mean is taken over time dimension
    try:
        ax.set_title(param_dict["title"])
        del param_dict["title"]
    except:
        pass

    mse = np.mean((data1 - data2) ** 2, axis=1)  # 250
    y_max = len(mse)
    y = np.arange(y_max)

    ax.plot(mse, y, **param_dict)
    ax.set_xlim(0, 0.03)
    ax.set_ylim(0, y_max)

    return ax
