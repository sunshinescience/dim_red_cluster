import matplotlib.pyplot as plt


def plot_images(data, reshape=None, save_fname=None, show=False):
    """
    Input reshape values as tuple of two numbers (e.g., (8, 8))
    """
    inversed_lst = range(0, 10)
    fig = plt.figure(figsize=(10,2))
    plt_index = 0
    for i in inversed_lst:
        plt_index = plt_index + 1
        ax = fig.add_subplot(1, 10, plt_index)
        data_ind = data[i]
        if reshape is not None:
            data_ind.reshape(reshape)
        ax.imshow(data_ind, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.tight_layout()
    if save_fname is not None:
        plt.savefig(save_fname, dpi=150)
    if show:
        plt.show()
