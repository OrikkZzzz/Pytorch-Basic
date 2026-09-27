import torchvision

#准备测试集
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_data = torchvision.datasets.CIFAR10("./datasetv2", train = False, transform=torchvision.transforms.ToTensor())

#batch_size:每次取四个数据集(mini-batch)
#shuffle:随机抓batch
#drop_last:数据集最后不足batch_size的是否舍去
test_loader = DataLoader(dataset=test_data, batch_size=4, shuffle=True, num_workers=0, drop_last=False)

#测试数据集中第一张图片以及target
img, target = test_data[0]
print(img.shape)
print(target)

writer = SummaryWriter("dataloader")
step = 0
for data in test_loader:
    imgs, targets = data
    writer.add_images("test_data", imgs, step)
    step = step + 1

writer.close()
