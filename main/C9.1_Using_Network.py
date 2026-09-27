import torch
from torch import nn
import torchvision
from torchvision.models import vgg16, VGG16_Weights

model = vgg16(weights=VGG16_Weights.DEFAULT)

model.add_module('add_linear', nn.Linear(1000,10))  # 修改现有模型（添加）
print(model)