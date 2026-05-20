"""
model.py - SE-CNN 滑坡易发性评价模型

模型架构：
    SE注意力模块（Squeeze-and-Excitation）+ 三层卷积 + 全连接分类器

    输入: [batch, 7, 11, 11] — 7个影响因子，11×11像素窗口
    输出: [batch, 2] — 二分类概率（非滑坡/滑坡）

网络结构：
    SE-Block → Conv2d(7→64) → BN → Tanh → MaxPool
    SE-Block → Conv2d(64→132) → BN → Tanh → MaxPool
    SE-Block → Conv2d(132→196) → BN → Tanh
    Flatten → Linear(784→384) → Linear(384→2) → Sigmoid
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from read_data import config


class SENetBlock(nn.Module):
    """
    Squeeze-and-Excitation (SE) 注意力模块

    通过全局平均池化获取通道级统计信息，再通过两层全连接网络
    学习通道间的依赖关系，对各通道进行自适应加权。

    参数：
        c: 输入通道数
        r: 压缩比率（默认16）
    """
    def __init__(self, c, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)


class LandslideCNN(nn.Module):
    """
    SE-CNN 滑坡易发性评价模型

    将 SE 注意力机制嵌入 CNN 的每一层卷积之前，
    使模型能够自适应地关注对滑坡预测最重要的影响因子通道。
    """
    def __init__(self):
        super(LandslideCNN, self).__init__()
        # SE注意力模块
        self.se1 = SENetBlock(config["feature"])
        self.se2 = SENetBlock(64)
        self.se3 = SENetBlock(132)

        # 卷积层
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=config["feature"], out_channels=64,
                      kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=132,
                      kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(132),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=132, out_channels=196,
                      kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(196),
            nn.Tanh(),
        )

        # 全连接分类器
        self.classifier = nn.Sequential(
            nn.Linear(784, 384),
            nn.Linear(384, 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        # SE注意力 → 卷积 → SE注意力 → 卷积 → SE注意力 → 卷积
        x = self.se1(x)
        out = self.conv_block1(x)

        out = self.se2(out)
        out = self.conv_block2(out)

        out = self.se3(out)
        out = self.conv_block3(out)

        # 展平 → 分类
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


if __name__ == "__main__":
    net = LandslideCNN()
    x = torch.randn(1, 7, 11, 11)
    out = net(x)
    print(f"输入尺寸: {x.shape}")
    print(f"输出尺寸: {out.shape}")
    print(f"输出值: {out}")
    print(f"\n模型参数量: {sum(p.numel() for p in net.parameters()):,}")
