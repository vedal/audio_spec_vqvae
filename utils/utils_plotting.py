import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

if os.environ.get('DISPLAY', '') == '':
    print('no display found. Using non-interactive Agg backend')
    matplotlib.use('Agg')


def plot_loss_with_annotated_min(ylist,
                                 epoch_offset=0,
                                 fontsize=10,
                                 save_path=None):

    fig, ax = plt.subplots(figsize=(10, 10))

    ax.plot(ylist)

    ymin = np.min(ylist)
    ymin_idx = np.argmin(ylist)
    mask = np.array(ylist) == ymin
    color = np.where(mask, 'red', 'white')
    s = np.where(mask, 100, 0)
    ax.scatter(range(len(ylist)), ylist, color=color, s=s)
    ax.annotate('Epoch: {}\nMSE: {:.4f}'.format(ymin_idx + epoch_offset, ymin),
                (ymin_idx, ymin),
                fontsize=fontsize)

    plt.title('Loss, testset', fontsize=20)
    plt.xlabel('Epochs', fontsize=15)
    plt.ylabel('MSE', fontsize=15)

    if save_path:
        fig.savefig(save_path + 'plot.eps', bbox_inches='tight')

    return


def plot_original_reconstruction(x_val,
                                 out,
                                 epoch,
                                 save_path=None,
                                 is_spectrogram=False):
    if is_spectrogram:
        from librosa.display import specshow

    fig, axes = plt.subplots(nrows=1, ncols=2)

    for (ax, title, image) in [(axes[0], 'Original', x_val),
                               (axes[1], 'Reconstruction', out)]:

        ax.set_title(title)

        if is_spectrogram:
            image = image[0, ..., 0].T
            specshow(image, ax=ax)
        else:
            image = image[0].astype(np.uint8)  # np.uint8
            ax.imshow(image)

    plt.tight_layout()

    if save_path:
        fig.savefig('{}epoch-{:04}.eps'.format(save_path, epoch),
                    bbox_inches='tight')


def plot_perplexity(perplexities, save_path=None):
    """Plot VQ-VAE cookbook perplexity graph
    
    Arguments:
        perplexities {list[float]} -- [perpl per iteration]
    
    Keyword Arguments:
        save_path {str} -- [if specified, save path] (default: {None})
    """
    plt.figure(figsize=(10, 10))
    plt.plot(perplexities)
    plt.title('Perplexity')
    plt.xlabel('Iterations')
    plt.ylabel('perplexity')
    fig = plt.gcf()
    if save_path:
        fig.savefig(save_path + 'perplexity.eps', bbox_inches='tight')


def close_plots():
    plt.close('all')
    return


# ---------------- for testing


def plot_loss_curves(d, save_path=None):
    fig, _ = plt.subplots(figsize=(15, 15))

    for k, v in d.items():
        plt.plot(v, label=k)

    plt.legend()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path + 'loss_curves.png', bbox_inches='tight')
    return
