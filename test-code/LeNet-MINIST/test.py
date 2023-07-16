import torchvision as tv
import torchvision.transforms as tfs

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

from model import LeNet_5

import os
import numpy as np

transform = tfs.Compose(
    [tfs.ToTensor()]
)

test_dataset = tv.datasets.MNIST(
    root="data",
    train=False,
    download=False,
    transform=transform
)

test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=4,
    shuffle=False
)

# 标签
classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

model = LeNet_5()

model.load_state_dict(torch.load("checkpoints/model_5.pth"))

if torch.cuda.is_available():
    model.cuda()

data_len = len(test_dataset)

corrent_num = 0
for i, data in enumerate(test_dataloader):
    inputs, labels = data
    if torch.cuda.is_available():
        inputs = inputs.cuda()
        labels = labels.cuda()
    inputs, labels = Variable(inputs), Variable(labels)

    outputs = model(inputs)

    _, predicted = torch.max(outputs, 1)

    for j in range(len(predicted)):
        predicted_num = predicted[j].item()
        label_num = labels[j].item()
        if predicted_num == label_num:
            corrent_num += 1

corrent_rate = corrent_num / data_len
print("correct rate is {:.3f}".format(corrent_rate * 100))
