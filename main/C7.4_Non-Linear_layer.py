import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from torch.nn import ReLU, Sigmoid

input = torch.tensor([[1,-0.5],
                      [-1,3]])

output = torch.reshape(input,(-1,1,2,2))
print(output.shape)

dataset = torchvision.datasets.CIFAR10("./datasetv2", train=False, download=True, transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64)

class MyNonLinear(nn.Module):
    def __init__(self):
        super(MyNonLinear,self).__init__()
        self.relu1 = ReLU()  #inplace:是否原地操作
        self.sigmoid = Sigmoid()

    def forward(self, input):
        output = self.sigmoid(input)
        return output

mymodel = MyNonLinear()

writer = SummaryWriter("./logs_relu")
step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, global_step=step)
    output = mymodel(imgs)
    writer.add_images("output", output, step)
    step += 1

writer.close()

