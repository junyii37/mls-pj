import argparse
import sys
from torchvision.datasets import MNIST, CIFAR10
from mynn.data import mnist_augment, preprocess, basic_mnist_augment, merge_datasets, cifar10_augment, basic_cifar10_augment
from mynn.layer import Flatten, Linear, ReLU, He, Conv, Dropout, Pooling, BN
from mynn.layer.blocks import BasicBlock
from mynn.loss import CrossEntropy
from mynn import Model
from mynn.optimizer import SGD, Adam, MomentGD
from mynn.runner import RunnerM, EarlyStopping, CosineAnnealingLR
import cupy as cp



def parse_args():
    parser = argparse.ArgumentParser(description="Train a CNN on MNIST with configurable hyperparameters")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.35, help="L2 weight decay coefficient")
    parser.add_argument("--rate", type=float, default=0.5, help="Dropout rate")
    parser.add_argument("--batch_size", type=int, default=256, help="Training batch size")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--T_max", type=int, default=25, help="T_max for CosineAnnealingLR")
    parser.add_argument("--eta_min", type=float, default=1e-6, help="eta_min for CosineAnnealingLR")
    parser.add_argument("--patience", type=int, default=15, help="Patience for EarlyStopping")
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
    train_dataset = CIFAR10(
        root="../dataset",
        train=True,
        transform=basic_cifar10_augment(train=True),
        download=True
    )
    test_dataset = CIFAR10(
        root="../dataset",
        train=False,
        transform=basic_cifar10_augment(train=False),
        download=True
    )



    train_images, train_labels = preprocess(train_dataset)
    test_images,  test_labels  = preprocess(test_dataset)

    train_set = (train_images[:45000], train_labels[:45000])
    dev_set   = (train_images[45000:], train_labels[45000:])
    test_set  = (test_images,       test_labels)

    # 数据增强
    train_dataset_augment = CIFAR10(
        root="../dataset",
        train=True,
        transform=cifar10_augment(train=True),
        download=True
    )

    train_images, train_labels = preprocess(train_dataset_augment)
    train_set_augment = (train_images[:45000], train_labels[:45000])
    dev_set_augment   = (train_images[45000:], train_labels[45000:])

    train_set = merge_datasets(train_set, train_set_augment)
    dev_set = merge_datasets(dev_set, dev_set_augment)

    def resnet_block(input_channels, num_channels, num_residuals, first_block=False, weight_decay=0.0):
        blk = []
        for i in range(num_residuals):
            if i== 0 and not first_block:
                blk.append(BasicBlock(input_channels, num_channels, downsampling=True, weight_decay=weight_decay))
            else:
                blk.append(BasicBlock(num_channels, num_channels, weight_decay=weight_decay))
        return blk

    # ResNet-18 近似实现
    layers = [
        # Conv1, no downsampling: (3, 32, 32) -> (16, 32, 32)
        Conv(in_channel=3, out_channel=16, kernel=5, stride=1, padding=2, weight_decay=args.weight_decay),
        BN(normalized_dims=(0, 2, 3), param_shape=(1, 16, 1, 1)),
        # Pooling(kernel=2),

        # Stage 1, no downsampling: (16, 32, 32) -> (16, 32, 32)
        *resnet_block(16, 16, 2, True, weight_decay=args.weight_decay),

        # Stage 2, downsampling: (16, 32, 32) -> (32, 16, 16)
        *resnet_block(16, 32, 2, weight_decay=args.weight_decay),

        # Stage 3, downsampling: (32, 16, 16) -> (64, 8, 8)
        *resnet_block(32, 64, 2, weight_decay=args.weight_decay),

        # Stage 4, downsampling: (64, 8, 8) -> (128, 4, 4)
        *resnet_block(64, 128, 2, weight_decay=args.weight_decay),

        # 最大值池化近似代替全局平均池化: (128, 4, 4) -> (128, 1, 1)
        Pooling(kernel=4),

        # 全连接层
        Flatten(),
        Dropout(rate=args.rate),
        Linear(in_channel=128, out_channel=10, weight_decay=args.weight_decay),
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

    # # 5. 模型参数可视化
    # model.visualize_parameters()


if __name__ == "__main__":
    main()
