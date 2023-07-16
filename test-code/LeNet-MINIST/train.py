import torchvision as tv
import torchvision.transforms as tfs

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable

from model import LeNet_5

import os
import numpy as np

# 对数据的预处理
transform = tfs.Compose(
    [tfs.ToTensor()]  # 转换为Tensor，并归一化至[0, 1]
)

# 训练集
train_dataset = tv.datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transform
)

train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=4,
    shuffle=True
)

# 标签
classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

net = LeNet_5()

if torch.cuda.is_available():
    net.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.9)

for epoch in range(5):
    running_loss = 0.0
    for i, data in enumerate(train_dataloader):
        inputs, labels = data
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()
        inputs, labels = Variable(inputs), Variable(labels)

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        # 每2000个batch打印一次训练状态
        if i % 100 == 99:
            print(
                "[{}/{}][{}/{}] loss:{:.3f}".format(epoch + 1, 5, (i + 1) * 4, len(train_dataset), running_loss / 100))

    torch.save(net.state_dict(), "checkpoints/model_{}.pth".format(epoch + 1))
    print("model_{}.pth saved".format(epoch + 1))

print("Finished Training!")
