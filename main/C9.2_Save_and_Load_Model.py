import torch
import torchvision

vgg16 = torchvision.models.vgg16(weights = torchvision.models.VGG16_Weights.DEFAULT)
# 保存+加载方式1
torch.save(vgg16, "vgg16_meth1.pth")
model1 = torch.load("vgg16_meth1.pth")

# 保存+加载方式2
torch.save(vgg16.state_dict(), "vgg16_meth2.pth")  # 保存为字典形式
model2 = torchvision.models.vgg16()
model2.load_state_dict(torch.load("vgg16_meth2.pth"))