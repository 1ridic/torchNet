import click
import os
from config import *
from model import *


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
