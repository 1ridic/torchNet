
import click
import os
from model import *

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


@click.command()
@click.option('--lr', type=float, default=0.01, help='学习率')
@click.option('--epoch', type=int, default=100, help='训练轮数')
@click.option('--patience', type=int, default=8, help='动态学习率耐心')
@click.option('--factor', type=float, default=0.2, help='动态学习率因子')
@click.option('--bs', type=int, default=80, help='batch_size')
def main(lr, epoch, patience, factor, bs):
    r = Runtime(layer)
    r.config(trainDataPath='./train/', testDataPath='./test/', modelSavePath='./output/',  # 配置路径
             testDataTransformer=img_transformer, trainDataTransformer=test_transformer,  # 配置预处理
             optimizer=optim.SGD(r.net.parameters(), lr=lr),  # 配置优化器
             lossFn=nn.CrossEntropyLoss(),  # 配置损失函数
             patience=patience, factor=factor)
    r.train(epoch, bs)


if __name__ == "__main__":
    main()
