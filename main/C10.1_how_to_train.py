from torch import nn
import torch
import torchvision
from torch.utils.data import DataLoader
from model import Mymodel
from torch.utils.tensorboard import SummaryWriter
 
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
            outputs = mymodel(imgs)
            loss = loss_function(outputs, targets)
            total_test_loss = total_test_loss + loss
    print("整体测试集上的Loss:{}".format(total_test_loss))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    total_test_step += 1

    torch.save(mymodel, "mymodel_{}.pth".format(i+1))

writer.close()
    
