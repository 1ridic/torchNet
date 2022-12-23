import torch.nn as nn  # 各种层类型的实现
import torch.nn.functional as F  # 各中层函数的实现，与层类型对应，如：卷积函数、池化函数、归一化函数等等
import torch.optim as optim  # 实现各种优化算法的包
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder  # 图形数据集
import torch.optim as optim  # pytorch优化工具箱
import os  # os文件系统工具包
import torch  # torch核心以依赖
import math
import cv2

# 图像预处理
img_transformer = transforms.Compose([
    # transforms.ToPILImage(),
    transforms.RandomRotation(5),  # 随机旋转
    transforms.Resize(128),  # 重设大小
    transforms.RandomResizedCrop(112, scale=(0.6, 1.0)),  # 随机裁剪 0.8-1.0倍
    # transforms.RandomHorizontalFlip(),#随机水平翻转
    transforms.ToTensor(),  # 转为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                         0.229, 0.224, 0.225])  # 标准化
])

test_transformer = transforms.Compose([
    # transforms.ToPILImage(),
    # transforms.RandomRotation(5), #随机旋转
    transforms.Resize(128),  # 重设大小
    transforms.RandomResizedCrop(112, scale=(0.6, 1.0)),  # 随机裁剪 0.8-1.0倍
    # transforms.RandomHorizontalFlip(),#随机水平翻转
    transforms.ToTensor(),  # 转为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                         0.229, 0.224, 0.225])  # 标准化
])

# 构造网络
layer = nn.Sequential(
    nn.Conv2d(
        in_channels=3,
        out_channels=16,
        kernel_size=3,
        stride=1,
        padding=1
    ),  # 维度变换(3,112,112) --> (16,112,112)
    nn.BatchNorm2d(num_features=16),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),  # 维度变换(16,112,112) --> (32,112,112)
    nn.Conv2d(
        in_channels=16,
        out_channels=32,
        kernel_size=3,
        stride=1,
        padding=1
    ),  # 维度变换(32,112,112) --> (32,56,56)
    nn.BatchNorm2d(num_features=32),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),  # 维度变换(32,56,56) --> (32,28,28)
    nn.Flatten(),  # 维度变换(32,28,28) --> (32*28*28,1)
    nn.Linear(32*28*28, 10),
    nn.Linear(10, 4)
)