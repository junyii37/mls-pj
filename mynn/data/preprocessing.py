import numpy as np
import os
import pickle


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

    if images.max() > 1.0:
        images = images / 255.0

    # transfer targets to one-hot labels
    labels = dataset.targets
    labels_onehot = np.zeros((len(labels), 10))
    for n, _class in enumerate(labels):
        labels_onehot[n][_class] = 1

    return images, labels_onehot


def load_cifar10_batch(filename):
    with open(filename, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
        images = batch[b'data'].reshape(-1, 3, 32, 32)
        labels = np.array(batch[b'labels'])
        labels_onehot = np.zeros((len(labels), 10))
        for n, label in enumerate(labels):
            labels_onehot[n][label] = 1

        return images, labels_onehot


def load_cifar10_dataset(data_dir):
    train_images = []
    train_labels = []
    for i in range(1, 6):
        batch_file = os.path.join(data_dir, f'data_batch_{i}')
        imgs, lbls = load_cifar10_batch(batch_file)
        train_images.append(imgs)
        train_labels.append(lbls)
    train_images = np.concatenate(train_images, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)

    # 加载测试集
    test_file = os.path.join(data_dir, 'test_batch')
    test_images, test_labels = load_cifar10_batch(test_file)

    return train_images/255.0, train_labels, test_images/255.0, test_labels
