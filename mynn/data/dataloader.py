import cupy as cp


def dataloader(dataset, batch_size, shuffle=True):
    """
    [Inputs]
        dataset: tuple (numpy.ndarray)
        batch_size: int
        shuffle: bool

    [Yields]
        batch_images, batch_labels: cupy.ndarray
    """
    images, labels = dataset

    # 在此处将 numpy 数组转换为 cupy 数组
    images = cp.array(images)
    labels = cp.array(labels)

    n = len(images)
    index = cp.arange(n)

    if shuffle:
        cp.random.shuffle(index)

    for i in range(0, n, batch_size):
        j = min(i + batch_size, n)
        yield images[index[i:j]], labels[index[i:j]]

