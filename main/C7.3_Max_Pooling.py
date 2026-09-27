#maxpooling：最大池化（下采样）降低特征维度，减少计算量，保留最重要特征 最常用
#maxunpooling:上采样
#floor:向下取整 ceiling：向上取整

import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("./datasetv2", train=False, download=True, transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=64)

class Mypooling(nn.Module):
    def __init__(self):
        super(Mypooling, self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=True)

    def forward(self, input):
        output = self.maxpool1(input)
        return output

pooling = Mypooling()

writer = SummaryWriter("logs_maxpool")
step = 0

for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, step)
    output = pooling(imgs)
    writer.add_images("output", output, step)
    step = step + 1

writer.close()

