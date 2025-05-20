import numpy as np
import os
import pickle

def preprocess(dataset):
    """
    Data Preprocessing

    [Inputs]
        dataset (torchvision.datasets)

    [Outputs]
        images (numpy.ndarray): shape (N, C, H, W), values in [0,1]
        labels_onehot (numpy.ndarray): one-hot encoded labels with shape (N, num_classes)
    """
    # 取出原始数据
    data = dataset.data
    # 如果是 PyTorch Tensor，就转为 NumPy
    if hasattr(data, 'numpy'):
        images = data.numpy()
    else:
        images = data

    # 处理通道顺序：
    # - MNIST: (N, H, W) -> (N, 1, H, W)
    # - CIFAR-10: (N, H, W, C) -> (N, C, H, W)
    if images.ndim == 3:
        # 灰度图
        images = images.reshape(-1, 1, images.shape[1], images.shape[2])
    elif images.ndim == 4:
        # 彩色图
        images = images.transpose(0, 3, 1, 2)
    else:
        raise ValueError(f"Unsupported image dimensions: {images.shape}")

    # 转为 float32 并归一化到 [0,1]
    images = images.astype(np.float32)
    if images.max() > 1.0:
        images /= 255.0

    # 构造 one-hot 标签
    labels = np.array(dataset.targets)
    num_classes = labels.max() + 1
    labels_onehot = np.zeros((len(labels), num_classes), dtype=np.float32)
    labels_onehot[np.arange(len(labels)), labels] = 1

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
