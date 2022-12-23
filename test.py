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
from PIL import Image
import sys
from config import *

class_names = ['j01', 'j03', 'j04', 'j10']

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python test.py <model_path> <img_path>")
        exit(1)
    model_path = sys.argv[1]
    img_path = sys.argv[2]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    print("Loading model...")
    net=torch.load(model_path)
    net=net.to(device)
    net.eval()

    print("Loading image...")
    Img_PIL = Image.open(img_path)
    Img_Sensor = test_transformer(Img_PIL)
    Img_Sensor = Img_Sensor.unsqueeze(0)
    Img_Sensor = Img_Sensor.to(device)
    
    out=net(Img_Sensor)
    _, pred = torch.sort(out, descending=True)
    percentage=F.softmax(out, dim=1)[0] * 100
    for i in range(4):
        print("Predicted: %s %.04f%%" % (class_names[pred[0][i]], percentage[pred[0][i]]))


