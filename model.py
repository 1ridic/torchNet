import time
import torch.nn as nn  # 各种层类型的实现
import torch.nn.functional as F  # 各中层函数的实现，与层类型对应，如：卷积函数、池化函数、归一化函数等等
import torch.optim as optim  # 实现各种优化算法的包
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder  # 图形数据集
import torch.optim as optim  # pytorch优化工具箱
import os  # os文件系统工具包
import torch  # torch核心以依赖

# 定义网络


class Net(nn.Module):
    def __init__(self, l):
        super(Net, self).__init__()
        self.layer = l

    def forward(self, x):
        logits = self.layer(x)
        return logits


class Runtime:
    def __init__(self, layer):
        self.t = time.strftime('%Y-%m-%d-%H-%M-%S')
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {self.device} device")

        self.net = Net(layer).to(self.device)
        print(self.net)

    def config(self, trainDataPath, trainDataTransformer, testDataPath, testDataTransformer, modelSavePath, lossFn, optimizer):
        self.train_data = ImageFolder(
            trainDataPath, transform=trainDataTransformer)
        print(self.train_data)
        print(self.train_data.class_to_idx)

        self.test_data = ImageFolder(
            testDataPath, transform=testDataTransformer)
        print(self.test_data)
        print(self.test_data.class_to_idx)

        self.modelSavePath = modelSavePath
        self.loss_fn = lossFn
        self.optimizer = optimizer

    def test(self):
        self.net.eval()
        correct = 0
        total = 0
        loader=DataLoader(dataset=self.test_data)
        total = len(loader.dataset)
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            with torch.no_grad():
                logits = self.net(x)
                pred = logits.argmax(dim=1)
            correct += torch.eq(pred, y).sum().float().item()
        return correct / total


    def saveModel(self,loss,acc):
        if not os.path.exists(self.modelSavePath):
            os.mkdir(self.modelSavePath)
        if not os.path.exists(self.modelSavePath + self.t):
            os.mkdir(self.modelSavePath + self.t)
        torch.save(self.net.state_dict(), self.modelSavePath + self.t + '/loss{:.4f}acc{:.4f}.pth'.format(loss,acc))

    def train(self,epoch,bs):
        best_loss = 0.0
        best_acc = 0.0
        for epoch in range(epoch):
            running_loss = 0.0
            self.net.train()
            for step,(features, targets) in enumerate(DataLoader(dataset=self.train_data, batch_size=bs, shuffle=True),0):
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

                    running_loss += loss.item()/bs

            test_acc=self.test()
            print('[%04d] loss: %02.04f%% | test_acc: %02.04f%%' % (epoch + 1, running_loss*100, test_acc*100))
            if(test_acc>best_acc):
                best_acc=test_acc
                self.saveModel(running_loss,test_acc)
            elif(running_loss<best_loss and test_acc==best_acc):
                best_loss=running_loss
                self.saveModel(running_loss,test_acc)
            # zero the loss
            running_loss = 0.0
