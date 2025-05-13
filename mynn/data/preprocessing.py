import numpy as np


def preprocess(dataset):
    """
    Data Preprocessing

    [Inputs]
        dataset (torchvision.datasets)

    [Outputs]
        images, labels_onehot (numpy.ndarray)
    """
    # 将数据转换为 numpy，并添加一个维度
    images = dataset.data.numpy().reshape(-1, 1, 28, 28)

    # transfer targets to one-hot labels
    labels = dataset.targets
    labels_onehot = np.zeros((len(labels), 10))
    for n, _class in enumerate(labels):
        labels_onehot[n][_class] = 1

    return images, labels_onehot
