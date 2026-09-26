from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter("logs")
img_path = "dataset\\dataset\\train\\ants\\5650366_e22b7e1065.jpg"
img_PIL = Image.open(img_path)
img_array = np.array(img_PIL)
print(type(img_array))

writer.add_image("test", img_array, 3, dataformats="HWC")       #从PIL到numpy需要在add_image中指定dataformats="HWC"；从PIL到tensor需要在add_image中指定dataformats="CHW"

for i in range(100):
    writer.add_scalar("y=2x", 2*i ,i)


writer.close()

# start /B tensorboard --logdir=logs --port=6006     /B:后台另一进程启动，不占用当前程序；--port=6006:指定端口号为6006，默认是6006

