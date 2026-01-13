import random
import torch
from torch.utils.data import Dataset, Subset
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict

transform_2 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # 归一化到 [-1, 1]
])
transform_3 = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化到 [-1, 1]
    transforms.Normalize((0.4914, 0.4822, 0.4465), 
                         (0.2023, 0.1994, 0.2010))
])

import time 
time_past = time.time()
# 下载训练数据和测试数据
full_dataset = torchvision.datasets.CIFAR10(root="/data/tianzhen/DATASETS/cifar10/raw_data", 
                                        train=True, download=False, transform=transform_3)
# FIXME
# testset = torchvision.datasets.CIFAR10(root="/data/tianzhen/DATASETS/cifar10/raw_data", 
#                                        train=False, download=False, transform=transform_3)
# trainset = Subset(trainset, list(range(10000)))
# testset = Subset(testset, list(range(10000)))

SEED = 2026

# 2. 定义划分比例
train_ratio = 0.8  # 训练集比例
test_ratio = 0.2   # 测试集比例

# 3. 计算各集合大小
train_size = int(train_ratio * len(full_dataset))
test_size = len(full_dataset) - train_size

# 4. 随机划分
trainset, testset = random_split(
    full_dataset, 
    [train_size, test_size],
    generator=torch.Generator().manual_seed(SEED)  # 设置随机种子保证可复现
)


import torch
from torch.utils.data import DataLoader, Subset
from collections import defaultdict
import random

def split_iid_data(trainset, num_clients, seed=SEED):
    """
    将MNIST训练集划分为指定客户端数量的IID数据
    确保每个客户端都包含所有类别，且每个类别的样本数尽可能均匀
    
    Args:
        trainset: MNIST训练数据集
        num_clients: 客户端数量
        seed: 随机种子
    
    Returns:
        client_datasets: 列表，每个元素是一个Subset，对应一个客户端的数据
    """
    # 设置随机种子
    random.seed(seed)
    torch.manual_seed(seed)
    
    # 按类别组织数据索引
    class_indices = defaultdict(list)
    for idx, (_, label) in enumerate(trainset):
        class_indices[label].append(idx)
    
    # 为每个类别打乱索引
    for label in class_indices:
        random.shuffle(class_indices[label])
    
    # 初始化每个客户端的数据索引
    client_indices = [[] for _ in range(num_clients)]
    
    # 对每个类别，将样本均匀分配到各个客户端
    for label, indices in class_indices.items():
        # 计算每个客户端应该分到的该类别样本数
        samples_per_client = len(indices) // num_clients
        remainder = len(indices) % num_clients
        
        start_idx = 0
        for client_id in range(num_clients):
            # 分配基础样本数
            end_idx = start_idx + samples_per_client
            
            # 将余数样本分配给前remainder个客户端
            if client_id < remainder:
                end_idx += 1
            
            # 将该类别的样本分配给客户端
            client_indices[client_id].extend(indices[start_idx:end_idx])
            start_idx = end_idx
    
    # 为每个客户端打乱数据（可选，但推荐）
    for client_id in range(num_clients):
        random.shuffle(client_indices[client_id])
    
    # 创建客户端数据集
    client_datasets = [Subset(trainset, indices) for indices in client_indices]
    
    return client_datasets
def split_non_iid_data(trainset, num_clients, alpha=0.5, seed=SEED):
    """
    使用狄利克雷分布将MNIST训练集划分为Non-IID数据
    
    Args:
        trainset: MNIST训练数据集
        num_clients: 客户端数量
        alpha: 狄利克雷分布参数，越小则Non-IID程度越高
        seed: 随机种子
    
    Returns:
        client_datasets: 列表，每个元素是一个Subset，对应一个客户端的数据
    """
    # 设置随机种子
    random.seed(seed)
    torch.manual_seed(seed)
    
    # 按类别组织数据索引
    class_indices = defaultdict(list)
    for idx, (_, label) in enumerate(trainset):
        class_indices[label].append(idx)
    
    # 为每个类别打乱索引
    for label in class_indices:
        random.shuffle(class_indices[label])
    
    # 初始化每个客户端的数据索引
    client_indices = [[] for _ in range(num_clients)]
    
    # 对每个类别进行狄利克雷分布采样和分配
    for label, indices in class_indices.items():
        # 从狄利克雷分布中采样，得到该类别的客户端分布比例
        proportions = torch.distributions.Dirichlet(
            torch.tensor([alpha] * num_clients)
        ).sample()
        
        # 根据比例计算每个客户端应该分到的样本数
        proportions = proportions * len(indices)
        client_samples = [int(p.item()) for p in proportions]
        
        # 处理由于取整可能导致的样本数不足
        total_allocated = sum(client_samples)
        remaining = len(indices) - total_allocated
        
        # 将剩余的样本分配给前几个客户端
        if remaining > 0:
            for i in range(remaining):
                client_samples[i % num_clients] += 1
        
        # 分配样本给各个客户端
        start_idx = 0
        for client_id in range(num_clients):
            end_idx = start_idx + client_samples[client_id]
            client_indices[client_id].extend(indices[start_idx:end_idx])
            start_idx = end_idx
    
    # 为每个客户端打乱数据
    for client_id in range(num_clients):
        random.shuffle(client_indices[client_id])
    
    # 创建客户端数据集
    client_datasets = [Subset(trainset, indices) for indices in client_indices]
    
    return client_datasets
def print_statistics(client_datasets, trainset):
    """打印数据分布统计信息"""
    num_clients = len(client_datasets)
    print(f"\n=== IID数据划分统计 ===")
    print(f"客户端数量: {num_clients}")
    print(f"总样本数: {len(trainset)}")
    print(f"每个客户端平均样本数: {len(trainset) // num_clients}")
    
    # 统计每个客户端的类别分布
    for client_id in range(min(3, num_clients)):  # 只显示前3个客户端
        client_indices = client_datasets[client_id].indices
        class_dist = defaultdict(int)
        
        for idx in client_indices:
            _, label = trainset[idx]
            class_dist[label] += 1
        
        print(f"\n客户端 {client_id} 统计:")
        print(f"总样本数: {len(client_indices)}")
        print("类别分布:", dict(sorted(class_dist.items())))
    
    # 总体统计
    total_samples = sum(len(client.indices) for client in client_datasets)
    print(f"\n总体统计:")
    print(f"分配的总样本数: {total_samples}")
    print(f"原始训练集样本数: {len(trainset)}")
def visualize_client_distribution(client_datasets, trainset, save_path=None, ignore_threshold=5):
    """
    可视化每个客户端的数据分布并保存图像
    
    Args:
        client_datasets: 客户端数据集列表
        trainset: 原始训练集（用于获取标签）
        num_clients: 客户端数量
        save_path: 图像保存路径，如果为None则不保存
        ignore_threshold: 忽略的差异阈值，小于此值的差异视为相同
    """
    # 设置中文字体（如果需要显示中文）
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    num_clients = len(client_datasets)
    # 统计每个客户端的类别分布
    client_distributions = []
    client_total_samples = []
    
    for client_id in range(num_clients):
        client_indices = client_datasets[client_id].indices
        class_dist = defaultdict(int)
        
        for idx in client_indices:
            _, label = trainset[idx]
            class_dist[label] += 1
        
        # 转换为有序列表（0-9）
        dist_array = [class_dist[i] for i in range(10)]
        client_distributions.append(dist_array)
        client_total_samples.append(len(client_indices))
    
    client_distributions = np.array(client_distributions)
    
    # 创建处理后的分布矩阵用于上色
    colored_distributions = np.zeros_like(client_distributions, dtype=float)
    
    # 对每个类别单独处理
    for class_idx in range(10):
        class_values = client_distributions[:, class_idx]
        
        # 如果所有值都为0，则跳过
        if np.all(class_values == 0):
            continue
            
        # 对非零值进行排序和分组
        unique_vals = np.unique(class_values)
        
        # 根据阈值对相似值进行分组
        groups = []
        current_group = [unique_vals[0]]
        
        for i in range(1, len(unique_vals)):
            if unique_vals[i] - current_group[-1] <= ignore_threshold:
                current_group.append(unique_vals[i])
            else:
                groups.append(current_group)
                current_group = [unique_vals[i]]
        groups.append(current_group)
        
        # 为每个组分配颜色值（基于组的大小顺序）
        group_colors = {}
        sorted_groups = sorted(groups, key=lambda x: np.mean(x))  # 按均值排序
        
        for i, group in enumerate(sorted_groups):
            color_value = i / max(len(sorted_groups) - 1, 1)  # 归一化到[0,1]
            for val in group:
                group_colors[val] = color_value
        
        # 应用颜色映射
        for client_idx in range(num_clients):
            val = class_values[client_idx]
            if val in group_colors:
                colored_distributions[client_idx, class_idx] = group_colors[val]
    
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 5))
    
    # 使用处理后的矩阵进行上色
    im = ax1.imshow(colored_distributions, cmap='YlOrRd', aspect='auto')
    ax1.set_xlabel('class', fontsize=12)
    ax1.set_ylabel('client ID', fontsize=12)
    title_suffix = f" (ignore_threshold≤{ignore_threshold})" if ignore_threshold > 0 else ""
    ax1.set_title(f'{num_clients} clients{title_suffix}', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(10))
    ax1.set_xticklabels([str(i) for i in range(10)])
    ax1.set_yticks(range(num_clients))
    ax1.set_yticklabels([f'Client {i}' for i in range(num_clients)])
    
    # 在热力图中显示原始数值
    for i in range(num_clients):
        for j in range(10):
            text = ax1.text(j, i, client_distributions[i, j],
                           ha="center", va="center", color="black", fontsize=8)
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label('Relative Size (per class)', rotation=270, labelpad=15)
    
    plt.tight_layout()
    
    # 保存图像
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图像已保存至: {save_path}")

cost_time = time.time()-time_past
print(f"dataset加载耗时: {cost_time:.2f}s")