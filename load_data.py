import numpy as np
import glob
import pickle

UNUSED_FILES = ['readme.html', 'batches.meta']


def unpickle(file):
    """
    param: file path
    return: decoded object
    """
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data


def load_data(dirs, negatives):
    data = None
    filenames, labels = list(), list()

    for i, dir in enumerate(dirs):
        train_dict = unpickle(dir)
        data = train_dict[b'data'] if i == 0 else np.vstack((data, train_dict[b'data']))
        filenames += train_dict[b'filenames']
        labels += train_dict[b'labels']

    data = data.reshape((len(data), 3, 32, 32))
    data = data.transpose(0, 2, 3, 1).astype(np.float32) if negatives else np.rollaxis(data, 1, 4)
    filenames = np.array(filenames)
    labels = np.array(labels)

    return data, filenames, labels


def load_CIFAR_10_data(data_dir, negatives=False):
    """
    params: data_dir: directory for the dataset
            negatives:

    return: data:
            filenames:
            labels:
    """

    files_names = glob.glob(data_dir + "/*")
    remove_additional = [''.join([data_dir, '/', i]) for i in UNUSED_FILES]
    _ = [files_names.remove(i) for i in remove_additional]

    return load_data(files_names, negatives)


if __name__ == "__main__":
    cifar_10_dir = 'cifar-10-batches-py'

    data, filenames, labels = load_CIFAR_10_data(cifar_10_dir, negatives=False)

    print("data: ", data.shape)
    print("filenames: ", filenames.shape)
    print("labels: ", labels.shape)
    print("labels are: ", set(labels))

    # Don't forget that the label_names and filesnames are in binary and need conversion if used.

    # display some random training images in a 25x25 grid
    # num_plot = 5
    # f, ax = plt.subplots(num_plot, num_plot)
    # for m in range(num_plot):
    #    for n in range(num_plot):
    #        idx = np.random.randint(0, data.shape[0])
    #        ax[m, n].imshow(data[idx])
    #        ax[m, n].get_xaxis().set_visible(False)
    #        ax[m, n].get_yaxis().set_visible(False)
    # f.subplots_adjust(hspace=0.1)
    # f.subplots_adjust(wspace=0)
    # plt.show()
