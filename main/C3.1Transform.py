from PIL import Image
from torchvision import transforms


#transform.py工具箱
#totensor/resize/...
#图片->工具->结果
#tensor数据类型：通过transforms.ToTensor()去解决两个问题
#1.transform如何使用
#2.tensor数据类型的优势

#绝对路径 D:\26~27Year\LabRotation\pytorch\dataset\dataset\train\ants\0013035.jpg
#相对路径 dataset\dataset\train\ants\0013035.jpg
img_path = "dataset\\dataset\\train\\ants\\0013035.jpg"
img = Image.open(img_path) 

tensor_trans = transforms.ToTensor() #将图片转为tensor数据类型
tensor_img = tensor_trans(img) #进行转换操作

print("img:", img)
print("tensor_img:", tensor_img)