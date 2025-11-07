
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # 归一化到 [-1, 1]
])

# 下载训练数据和测试数据
trainset = torchvision.datasets.MNIST(root="/data/tianzhen/DATASETS", train=True, download=False, transform=transform)
testset = torchvision.datasets.MNIST(root="/data/tianzhen/DATASETS", train=False, download=False, transform=transform)

# 数据加载器（批量读取）
train_loader = DataLoader(trainset, batch_size=64, shuffle=True)
test_loader = DataLoader(testset, batch_size=64, shuffle=False)
