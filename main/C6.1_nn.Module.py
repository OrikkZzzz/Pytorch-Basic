# https://docs.pytorch.org/docs/2.14/nn.html

import torch
from torch import nn

class MyNN(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self,input):
        output = input + 1
        return output

mynuern = MyNN()
x = torch.tensor(1.0)
output = mynuern(x)
print(output)