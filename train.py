import argparse
import sys
from torchvision.datasets import MNIST
from mynn.data import mnist_augment, preprocess, basic_mnist_augment, merge_datasets
from mynn.layer import Flatten, Linear, ReLU, He, Conv, Dropout, Pooling, BN
from mynn.loss import CrossEntropy
from mynn import Model
from mynn.optimizer import SGD, Adam, MomentGD
from mynn.runner import RunnerM, EarlyStopping, CosineAnnealingLR
import cupy as cp



def parse_args():
    parser = argparse.ArgumentParser(description="Train a CNN on MNIST with configurable hyperparameters")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.2, help="L2 weight decay coefficient")
    parser.add_argument("--rate", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--batch_size", type=int, default=256, help="Training batch size")
    parser.add_argument("--num_epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--T_max", type=int, default=10, help="T_max for CosineAnnealingLR")
    parser.add_argument("--eta_min", type=float, default=1e-6, help="eta_min for CosineAnnealingLR")
    parser.add_argument("--patience", type=int, default=7, help="Patience for EarlyStopping")
    parser.add_argument("--delta", type=float, default=0.0001, help="Minimum change to qualify as improvement for EarlyStopping")
    parser.add_argument("--shuffle", dest="shuffle", action="store_true", help="Enable data shuffling")
    parser.add_argument("--no-shuffle", dest="shuffle", action="store_false", help="Disable data shuffling")
    parser.set_defaults(shuffle=True)
    parser.add_argument("--save_dir", type=str, default="saved_models", help="Directory to save models and logs")
    parser.add_argument("--augment", type=bool, default=True, help="Enable data augmentation")
    return parser.parse_args()


def main():
    cp.random.seed(42)

    args = parse_args()
    print("Hyperparameters:", vars(args), file=sys.stdout)

    # 1. 数据加载与预处理
    train_dataset = MNIST(
        root="dataset",
        train=True,
        transform=basic_mnist_augment(train=True),
        download=True
    )
    test_dataset = MNIST(
        root="dataset",
        train=False,
        transform=basic_mnist_augment(train=False),
        download=True
    )



    train_images, train_labels = preprocess(train_dataset)
    test_images,  test_labels  = preprocess(test_dataset)

    train_set = (train_images[:50000], train_labels[:50000])
    dev_set   = (train_images[50000:], train_labels[50000:])
    test_set  = (test_images,       test_labels)

    # 数据增强
    train_dataset_augment = MNIST(
        root="dataset",
        train=True,
        transform=mnist_augment(train=True),
        download=True
    )

    train_images, train_labels = preprocess(train_dataset_augment)
    train_set_augment = (train_images[:50000], train_labels[:50000])
    dev_set_augment   = (train_images[50000:], train_labels[50000:])

    if True:
        train_set = merge_datasets(train_set, train_set_augment)
        dev_set = merge_datasets(dev_set, dev_set_augment)


    # 2. 模型结构

    # # Test 1: MLP
    # layers = [
    #     Flatten(),
    #
    #     # 第一层：宽度与初始化策略
    #     # Linear(28 * 28, 512),
    #     Linear(28 * 28, 512, weight_decay=0.2),
    #     ReLU(),
    #     Dropout(0.3),
    #
    #     # 第二层：深度扩展
    #     # Linear(512, 256),
    #     Linear(512, 256, weight_decay=0.2),
    #     ReLU(),
    #     Dropout(0.3),
    #
    #     # 第三层：特征压缩
    #     # Linear(256, 128),
    #     Linear(256, 128, weight_decay=0.2),
    #     ReLU(),
    #     Dropout(0.3),
    #     # 输出层
    #     Linear(128, 10),
    #     # Linear(128, 10, weight_decay=0.2),
    # ]

    # Test 2
    layers = [
        Conv(in_channel=1, out_channel=32, kernel=3, stride=1, padding=1, weight_decay=args.weight_decay),
        BN(normalized_dims=(0, 2, 3), param_shape=(1, 32, 1, 1), weight_decay=args.weight_decay),
        ReLU(),
        Pooling(kernel=2),

        Conv(in_channel=32, out_channel=64, kernel=3, stride=1, padding=1, weight_decay=args.weight_decay),
        BN(normalized_dims=(0, 2, 3), param_shape=(1, 64, 1, 1), weight_decay=args.weight_decay),
        ReLU(),
        Pooling(kernel=2),

        Flatten(),
        Dropout(rate=args.rate),

        Linear(in_channel=64 * 7 * 7, out_channel=10, weight_decay=args.weight_decay),
    ]

    model     = Model(layers)
    optimizer = Adam(model=model, lr=args.lr)
    # optimizer = SGD(model=model, lr=0.001)
    # optimizer = MomentGD(model=model, lr=0.001)
    loss_fn   = CrossEntropy(model=model)
    runner    = RunnerM(model=model, loss=loss_fn, optimizer=optimizer)

    # 3. 训练
    runner.train(
        train_set=train_set,
        dev_set=dev_set,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        scheduler=CosineAnnealingLR(
            optimizer=optimizer,
            T_max=args.T_max,
            eta_min=args.eta_min,
        ),
        strategy=EarlyStopping(
            patience=args.patience,
            delta=args.delta,
        ),
        shuffle=True,
        save_dir=args.save_dir
    )

    # 4. 测试集评估
    test_loss, test_acc = runner.evaluate(test_set, batch_size=args.batch_size)
    print(f"Test loss: {test_loss:.5f}, Test accuracy: {test_acc:.5f}")

    # 5. 模型参数可视化
    model.visualize_parameters()


if __name__ == "__main__":
    main()
