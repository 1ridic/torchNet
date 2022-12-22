import torch
import torch.nn as nn	# 各种层类型的实现
import torch.nn.functional as F	# 各中层函数的实现，与层类型对应，如：卷积函数、池化函数、归一化函数等等
import torch.optim as optim	# 实现各种优化算法的包
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
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
    def __init__(self,layer):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {self.device} device")

        self.net = Net(layer).to(self.device)
        print(self.net)

    def loadTrainData(self, path, transformer):
        self.train_data=ImageFolder(path, transform=transformer)
        print(self.train_data)
        print(self.train_data.class_to_idx)

    def train(self,epoch):
        self.net.train()
        for epoch in range(epoch):
            running_loss = 0.0
            running_acc = 0.0
            for step,(features, targets) in enumerate(DataLoader(dataset=self.train_data, batch_size=1, shuffle=True),0):
                    features = features.to(self.device)
                    targets = targets.to(self.device)
                    # 梯度清零，也就是把loss关于weight的导数变成0.
                    # 进⾏下⼀次batch梯度计算的时候，前⼀个batch的梯度计算结果，没有保留的必要了。所以在下⼀次梯度更新的时候，先使⽤optimizer.zero_grad把梯度信息设置为0。
                    self.optimizer.zero_grad()
                    # 获取网络输出
                    output = self.net(features)
                    # 获取损失
                    loss = self.loss_fn(output, targets)
                    # 反向传播
                    loss.backward()
                    # 训练
                    self.optimizer.step()

                    running_loss += loss.item()
 
            print('[%04d] loss: %03d' %
                      (epoch + 1,running_loss))
            # zero the loss
            running_loss = 0.0

        # # Compute and print the average accuracy fo this epoch when tested over all 10000 test images
        # accuracy = self.testAccuracy()
        
        # # we want to save the model if the accuracy is the best
        # if accuracy > best_accuracy:
        #     saveModel()
        #     best_accuracy = accuracy
            