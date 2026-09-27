from turtle import forward

import torchvision
import torch
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("./datasetv2", 
                                       train=False, 
                                       download=True, 
                                       transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=64)

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear1 = Linear(196608, 10)

    def forward(self, input):
        output = self.linear1(input)
        return output

mymodel = MyModel()

for data in dataloader:
    imgs, targets = data
    output = torch.flatten(imgs)