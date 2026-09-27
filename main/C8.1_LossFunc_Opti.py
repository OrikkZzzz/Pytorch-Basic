from torch import nn
import torch
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torchvision

dataset = torchvision.datasets.CIFAR10("D:/26~27Year/LabRotation/pytorch/datasetv2", train=False, 
                                       transform=torchvision.transforms.ToTensor(), download=True)

dataloader = DataLoader(dataset, Batch_size=1)

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = Conv2d(3, 32, 5, padding=2)
        self.maxpool1 = MaxPool2d(2)
        self.conv2 = Conv2d(32, 32, 5, padding=2)
        self.maxpool2 = MaxPool2d(2)
        self.conv3 = Conv2d(32, 64, 5, padding=2)
        self.maxpool3 = MaxPool2d(2)
        self.flatten = Flatten()
        self.linear1 = Linear(1024, 64)
        self.linear2 = Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.maxpool3(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)
        return x

# 这里用self.model1 = Sequential(...)效果是一样的

loss = nn.CrossEntropyLoss()

mymodel = MyModel()
optim = torch.optim.SGD(mymodel.parameters(), lr=0.01, )

for data in dataloader:
    imgs, targets = data
    outputs = mymodel(imgs)
    result_loss = loss(outputs, targets)
    optim.zero_grad()
    result_loss.backward()
    optim.step()
    