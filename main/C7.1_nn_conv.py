import torch
import torch.nn.functional as F

input = torch.tensor([[1, 2, 3, 0, 1],
                      [0, 1, 3, 5, 6],
                      [3, 1, 3, 0, 0],
                      [3, 6, 0, 2, 1],
                      [3, 4, 4, 7, 1]])

kernel = torch.tensor([[1, 1, 1],
                       [1, 2, 1],
                       [0, 1, 0]])

input = torch.reshape(input, (1, 1, 5, 5))
kernel = torch.reshape(kernel, (1, 1, 3, 3))

# print(input.shape)
# print(kernel.shape)

output = F.conv2d(input, kernel, stride=1)   #stride:步幅 padding:给input扩展一圈- dilation:空洞卷积
print(output)

