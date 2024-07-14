import torch
import torch.nn as nn


class ImprovedCNN(nn.Module):
    def __init__(self, num_classes):
        super(ImprovedCNN, self).__init__()

        # 定义神经网络模型
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  # 输入通道3 (RGB)，输出通道64，卷积核大小3x3，填充1
            nn.BatchNorm2d(64),  # 批量归一化，减少内部协变量偏移
            nn.ReLU(inplace=True),  # 激活函数，增加非线性
            nn.MaxPool2d(2),  # 最大池化，减少数据尺寸

            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 输入通道64，输出通道128，卷积核大小3x3，填充1
            nn.BatchNorm2d(128),  # 批量归一化
            nn.ReLU(inplace=True),  # 激活函数
            nn.MaxPool2d(2),  # 最大池化

            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # 输入通道128，输出通道256，卷积核大小3x3，填充1
            nn.BatchNorm2d(256),  # 批量归一化
            nn.ReLU(inplace=True),  # 激活函数
            nn.MaxPool2d(2),  # 最大池化

            nn.Conv2d(256, 512, kernel_size=3, padding=1),  # 输入通道256，输出通道512，卷积核大小3x3，填充1
            nn.BatchNorm2d(512),  # 批量归一化
            nn.ReLU(inplace=True),  # 激活函数
            nn.MaxPool2d(2),  # 最大池化

            nn.Flatten(),  # 展平操作，将多维张量展平成一维张量
            nn.Linear(512 * 4 * 4, 1024),  # 全连接层，输入特征维度512*4*4，输出特征维度1024
            nn.ReLU(inplace=True),  # 激活函数
            nn.Dropout(0.5),  # Dropout，防止过拟合
            nn.Linear(1024, num_classes)  # 全连接层，输出特征维度为类别数
        )

    def forward(self, x):
        x = self.model(x)
        return x
