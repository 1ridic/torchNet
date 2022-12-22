import torch
import torch.nn as nn	# 各种层类型的实现
import torch.nn.functional as F	# 各中层函数的实现，与层类型对应，如：卷积函数、池化函数、归一化函数等等
import torch.optim as optim	# 实现各种优化算法的包
from torchvision import datasets, transforms

from torchvision.datasets import ImageFolder  # 图形数据集
import torch.optim as optim  # pytorch优化工具箱
import os  # os文件系统工具包
import torch  # torch核心以依赖

## 定义网络
class Net(nn.Module):
    def __init__(self,l):
        super(Net, self).__init__()
        self.layer = l
    def forward(self, x):
        logits = self.layer(x)
        return logits

class Runtime:
    def __init__(self,l):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {self.device} device")
        self.model = Net(l).to(self.device)
        print(self.model)
    def loadTrainData(self, path, transformer):
        train_data=ImageFolder(path, transform=transformer)
        print(train_data)
        print(train_data.class_to_idx)