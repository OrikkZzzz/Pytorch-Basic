from torch import nn
import torch
import torchvision
from PIL import Image

img_path = "./datasetv2/train/0_0.png"
img = Image.open(img_path)

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32, 32)),
    torchvision.transforms.ToTensor()
])
img = transform(img)
print(img.shape)  # torch.Size([3, 32, 32])

#搭建神经网络
class Mymodel(nn.Module):
    def __init__(self):
        super(Mymodel, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,5,1,2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x

model = torch.load("./model.pth")

img = torch.reshape(img, (1, 3, 32, 32))
model.eval()                                    # 将模型转化为测试类型
with torch.no_grad():
    output = model(img)
