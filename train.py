import cv2
import os
from model import *

layer=nn.Sequential( # 构造网络
            nn.Flatten(),
            nn.Conv2d(64*64*3, 256, 5),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 5),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 5),
            nn.MaxPool2d(2),
            nn.Linear(32, 16),
            nn.Linear(16, 4)
        )
img_transformer = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(75),#重设大小
    transforms.RandomResizedCrop(64,scale=(0.8,1.0)),#随机裁剪 0.8-1.0倍
    # transforms.RandomHorizontalFlip(),#随机水平翻转
    transforms.ToTensor(),#转为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])#标准化
])
r=Runtime(layer)

r.loadTrainData('./train/', img_transformer)
##numpy.ndarray
# img = cv2.imread(img_path)  # 读取图像
# img1 = train_transformer(img)