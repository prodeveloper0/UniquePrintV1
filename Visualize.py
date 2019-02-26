import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def sample2fig(samples, rows, cols):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(rows, cols)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        sample = sample.reshape((sample.shape[0], sample.shape[1]))
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample, cmap='Greys_r')
    return fig


def savefig(fig, filename):
    fig.savefig(filename, bbox_inches='tight')
    plt.close(fig)


def showfig(fig):
    plt.show(fig)
    plt.close(fig)
