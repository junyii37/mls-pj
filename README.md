# 机器学习系统期末项目

本项目是机器学习系统期末项目的基础框架，从零开始构建了 **卷积** 神经网络，在 MNIST 数据集上进行图像分类任务。这篇文档主要介绍了如何训练及测试模型。



## 1. Introduction

模型被封装到 `mynn` 包中，包下含有 `data`、`layer`、`loss`、`optimizer`、`runner` 五个模块。`data` 模块中包含 DataLoader 、数据预处理和数据增强部分；`layer` 模块中包含涉及到了各个网络层和各种可选择激活函数；`loss` 模块中包含结合 Softmax 实现的交叉熵损失层；`optimizer` 中包含 SGD、MGD 和 Adam 优化器；`runner` 模块中包含所用到的早停策略、余弦退火调度策略以及运行器类 `runner`。运行器类能够自动训练模型、计算损失和精度、绘制曲线、根据实验时间自动创建文件夹并保存实验结果和模型参数。 `model` 包中还含有 `model.py` 类文件，定义了网络层的组织方式和模型的方法（训练、更新、预测、保存参数、读取参数等等）。

代码结构如下：
```
mls-pj/
├── dataset/
├── mynn/
│   ├── data/
│   ├── layer/
│   ├── loss/
│   ├── optimizer/
│   ├── runner/
│   ├── __init__.py
│   └── model.py
├── saved_models/
│   └── best_model/ 
├── .gitignore
├── README.md
├── test.py
└── train.py
```



## 2. Setup

首先需要将模型代码下载到本地：（$ 表示命令行提示符，复制时可忽略此符号）

```
$git clone https://github.com/junyii37/mls-pj.git
```

由于预训练模型较小，故已经放在 `saved_models/best_model` 文件夹中。如若需要，可以从 https://drive.google.com/file/d/1AMRUhEXBbhZ8ilb7YllgxNLihNBSQJjK/view?usp=sharing 下载预训练模型参数，并将其放在上述文件夹中，位置如前所示。

虽然代码在运行过程中会自动下载数据集，但如若需要，可以从 https://drive.google.com/file/d/1ITfA8eHALG3yxvHIvCzHZ97Gn_DvEx3r/view?usp=sharing 下载数据集，并将解压后的文件放在 `dataset` 文件夹中，位置如前所示。



## 3. Train

在项目文件夹路径下打开终端，输入如下指令，即可以默认配置进行训练，并进行参数可视化：

```
$python train.py
```

如果想要自定义超参数，可以先使用如下命令查看可选超参：

```
$python train.py --help
```

会显示如下可选择超参数：

```
options:
  -h, --help            show this help message and exit
  --lr LR               Learning rate
  --weight_decay WEIGHT_DECAY
                        L2 weight decay coefficient
  --rate RATE           Dropout rate
  --batch_size BATCH_SIZE
                        Training batch size
  --num_epochs NUM_EPOCHS
                        Number of training epochs
  --T_max T_MAX         T_max for CosineAnnealingLR
  --eta_min ETA_MIN     eta_min for CosineAnnealingLR
  --patience PATIENCE   Patience for EarlyStopping
  --delta DELTA         Minimum change to qualify as improvement for EarlyStopping
  --shuffle             Enable data shuffling
  --no-shuffle          Disable data shuffling
  --save_dir SAVE_DIR   Directory to save models and logs
  --augment AUGMENT     Enable data augmentation
```

可以据此添加可选参数项，例如：

```
$python train.py --lr 0.0001 --batch_size 256 --epochs 100 --patience 7
```



注意：如果想要更改模型架构，可以在 `train.py` 中修改 `layers` 部分即可。可选用的网络层有：

```
Conv, Linear, BN, ReLU, Pooling, Flatten, Dropout, Loss, LeakyReLU, Sigmoid, Softmax, Tanh
```

仿照文件中的代码组织方式，自定义网络架构即可。当然也可以修改优化器、调度算法、学习率调度策略。修改完成后按照第一步运行代码即可。训练好的模型会自动保存到 `saved_models` 文件夹中，以训练开始的系统时间命名文件夹。



## 4. Test

在项目文件夹路径下打开终端，输入如下指令，即可以默认模型参数路径（`saved_models/best_model/best_model.pickle`）进行训练：

```
$python test.py
```

如果想要自定义模型参数路径，可以使用如下命令（ `--model_path` 后面的路径为可更改自定义路径）：


## 5. attack and train

在此基础上，我们添加了attack方法，他们被封装在mynn.attack中，包括FGSM,BIM，PGD。
我们可以使用runner中的train_with_attack和train_with_trades进行相关对抗训练，从而提升模型鲁棒性。

## 6. attack and train result

我们将训练结果放在了attack文件夹和curve文件夹下，可以看到不同攻击方法的结果以及对抗训练带来的提升。


```



$python test.py --model_path 'saved_models/best_model/best_model.pickle'
```
