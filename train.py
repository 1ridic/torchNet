import cv2
import os
from model import *

# 图像预处理
img_transformer = transforms.Compose([
    # transforms.ToPILImage(),
    transforms.RandomRotation(5), #随机旋转
    transforms.Resize(128),#重设大小
    transforms.RandomResizedCrop(112,scale=(0.6,1.0)),#随机裁剪 0.8-1.0倍
    # transforms.RandomHorizontalFlip(),#随机水平翻转
    transforms.ToTensor(),#转为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])#标准化
])

test_transformer = transforms.Compose([
    # transforms.ToPILImage(),
    # transforms.RandomRotation(5), #随机旋转
    transforms.Resize(128),#重设大小
    transforms.RandomResizedCrop(112,scale=(0.6,1.0)),#随机裁剪 0.8-1.0倍
    # transforms.RandomHorizontalFlip(),#随机水平翻转
    transforms.ToTensor(),#转为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])#标准化
])

# 构造网络
layer=nn.Sequential( 
    nn.Conv2d(
        in_channels=3,
        out_channels=16,
        kernel_size=3,
        stride=1,
        padding=1
        ),                               #维度变换(3,112,112) --> (16,112,112)
    nn.BatchNorm2d(num_features=16),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),         #维度变换(16,112,112) --> (32,112,112)
    nn.Conv2d(
        in_channels=16,
        out_channels=32,
        kernel_size=3,
        stride=1,
        padding=1
            ),                           #维度变换(32,112,112) --> (32,56,56)
    nn.BatchNorm2d(num_features=32),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),        #维度变换(32,56,56) --> (32,28,28)
    nn.Flatten(),                       #维度变换(32,28,28) --> (32*28*28,1)
    nn.Linear(32*28*28, 10),
    nn.Linear(10, 4)
)


r=Runtime(layer)
r.config(trainDataPath='./train/', testDataPath='./test/', modelSavePath='./output/',  #配置路径
        testDataTransformer=img_transformer, trainDataTransformer=test_transformer,     #配置预处理
        optimizer=optim.SGD(r.net.parameters(), lr=0.01),                              #配置优化器
        lossFn=nn.CrossEntropyLoss())                                                  #配置损失函数
epoch=300
batch_size=100
r.train(epoch, batch_size)