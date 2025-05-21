import argparse
from pathlib import Path

from torchvision.datasets import MNIST, CIFAR10

from mynn.data import mnist_augment, preprocess, basic_cifar10_augment
from mynn import Model
from mynn.loss import CrossEntropy
from mynn.runner import RunnerM
import cupy as cp



def find_latest_model(saved_dir: Path) -> Path:
    """
    返回 saved_dir 下按名称排序最新的子文件夹中的 best_model.pickle
    """
    subs = [d for d in saved_dir.iterdir() if d.is_dir()]
    if not subs:
        raise FileNotFoundError(f"No subdirectories in {saved_dir}")
    latest = max(subs, key=lambda d: d.name)
    candidate = latest / "best_model.pickle"
    if not candidate.is_file():
        raise FileNotFoundError(f"No best_model.pickle in {latest}")
    return candidate


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load a trained model and evaluate on MNIST test set"
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        # default=Path("saved_models/2025-05-21_07-03-36/best_model.pickle"),
        default=None,
        help="Path to best_model.pickle (overrides --saved_dir lookup)"
    )
    parser.add_argument(
        "--saved_dir",
        type=Path,
        default=Path("../saved_models"),
        help="Directory in which to auto-detect the latest model subfolder"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for evaluation"
    )
    return parser.parse_args()


def main():
    cp.random.seed(42)

    args = parse_args()

    # 1. 定位模型文件
    if args.model_path is None:
        model_path = find_latest_model(args.saved_dir)
    else:
        model_path = args.model_path
    print(f"> Loading model from: {model_path}")

    # 2. 加载测试集并预处理
    test_ds = CIFAR10(
        root="../dataset",
        train=False,
        transform=basic_cifar10_augment(train=False),
        download=False
    )
    test_images, test_labels = preprocess(test_ds)
    test_set = (test_images, test_labels)

    # 3. 重建并加载模型
    model = Model().load_model(str(model_path))

    # 4. 构造损失层与 Runner（不用 optimizer）
    loss_fn = CrossEntropy(model=model)
    runner = RunnerM(model=model, loss=loss_fn, optimizer=None)

    # 5. 评估并输出
    test_loss, test_acc = runner.evaluate(test_set, batch_size=args.batch_size)
    print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
