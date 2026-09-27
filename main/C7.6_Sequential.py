import pytorch
from torch import nn

model = nn.Sequential(
            nn.Conv2d(1,20,5),
            nn.ReLU,
            nn.Conv2d(20,64,5),
            nn.ReLU
        )
# 等价于手动写：
# def forward(self, x):
#     x = self.layer0(x)
#     x = self.layer1(x)
#     x = self.layer2(x)
#     return x
# nn.Sequential = 按顺序打包多个层，并自动实现前向传播。适合简单线性堆叠；
# 如果网络有分支、循环、跳跃连接，就用自定义 nn.Module + nn.ModuleList / nn.ModuleDict