
import torch
from torch import nn


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)  # 输入层
        self.fc2 = nn.Linear(128, 64)     # 隐藏层
        self.fc3 = nn.Linear(64, 10)      # 输出层

    def forward(self, x):
        x = x.view(-1, 28*28)  # 展平28x28的图像
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
