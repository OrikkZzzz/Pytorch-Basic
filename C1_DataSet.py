#垃圾（数据）
#dataset：将某种垃圾拿出来并且能编号，获取label；提供一种方式去获取数据及其label
#如何获取每一个数据及其label
#告诉我们总共有多少数据
#dataloader：打包（0，1，2，3）；为后面的网络提供不同的数据形式

from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image

class MyData(Dataset):

    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path)

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.path, self.label_dir, img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label

    def __len__(self):
        return len(self.img_path)

root_dir = "dataset/dataset/train"
ants_label_dir = "ants"
ants_dataset = MyData(root_dir, ants_label_dir)