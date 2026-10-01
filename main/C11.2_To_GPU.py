from torch import nn
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 定义训练设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
# 准备数据集
train_data = torchvision.datasets.CIFAR10(root="./datasetv2", train=True, 
                                          transform=torchvision.transforms.ToTensor(),
                                          download=True)

test_data = torchvision.datasets.CIFAR10(root="./datasetv2", train=False, 
                                          transform=torchvision.transforms.ToTensor(),
                                          download=True)

#获取lenth长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集的长度为{}".format(train_data_size))
print("测试数据集的长度为{}".format(test_data_size))

#利用DataLoader获取数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

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

# 创建网络模型
mymodel = Mymodel()

# 损失函数
loss_function = nn.CrossEntropyLoss()

# 优化器
learning_rate = 0.01
optimizer = torch.optim.SGD(mymodel.parameters(), lr=learning_rate)

# 网络的一些参数
total_train_step = 0    #训练次数
total_test_step = 0     #测试次数
epoch = 10              #训练轮数

# add tensorboard
writer = SummaryWriter("./logs_train")

for i in range(epoch):
    print("-----------第{}轮训练开始-----------".format(i+1))
    for data in train_dataloader:
        imgs, targets = data
        imgs, targets = imgs.to(device), targets.to(device)  # 将数据移动到GPU上
        outputs = mymodel(imgs)
        loss = loss_function(outputs, targets)

        #优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print("训练次数：{}, Loss:{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    #测试步骤
    total_test_loss = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs, targets = imgs.to(device), targets.to(device)  # 将数据移动到GPU上
            outputs = mymodel(imgs)
            loss = loss_function(outputs, targets)
            total_test_loss = total_test_loss + loss
    print("整体测试集上的Loss:{}".format(total_test_loss))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    total_test_step += 1

    torch.save(mymodel, "mymodel_{}.pth".format(i+1))

writer.close()

