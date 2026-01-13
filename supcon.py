import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.fedarenax import get_resnet18, get_resnet18_supcon, get_train_loader, get_test_loader, SelfDefLoss,InfoNCELoss,SupConLoss
import time

def train_with_supcon(model, device, data_loader, optimizer, ce_criterion, supcon_criterion, num_epochs=1, round_single=0, rounds=1):
    model = model.to(device)
    model.train()
    log_str = ""
    for epoch in range(num_epochs):
        for idx, (data, labels) in enumerate(data_loader, start=1):
            data, labels = data.to(device), labels.to(device)
            if idx % 100 == 0:
                print(f"  Batch {idx}")
                log_str += f"  Batch {idx}\n"

            data = data.float() # 图像数据标准化
            labels = labels.long()

            # 梯度清零
            optimizer.zero_grad()

            # 前向传播
            features, logits = model(data)
            
            # loss = criterion(outputs, labels)
            ce_loss = ce_criterion(logits, labels) 
            supcon_loss = supcon_criterion(features, labels)
            loss = ce_loss + supcon_loss

            # 反向传播
            loss.backward()

            # torch.nn.utils.clip_grad_norm_(
            #     model.parameters(), 
            #     max_norm=10.0,  # 最大范数阈值
            #     norm_type=2    # L2范数
            # )

            # 更新参数
            optimizer.step()
            
        print(f"Epoch {epoch+1+round_single}/{num_epochs+rounds-1} {loss.item()}")
        log_str += f"Epoch {epoch+1+round_single}/{num_epochs+rounds-1} {loss.item()}\n"
    return log_str

def train(model, device, data_loader, optimizer, scheduler, criterion, num_epochs=1, round_single=0, rounds=1):
    model = model.to(device)
    model.train()
    for epoch in range(num_epochs):
        for idx, (data, labels) in enumerate(data_loader, start=1):
            data, labels = data.to(device), labels.to(device)
            if idx % 100 == 0:
                print(f"  Batch {idx}")

            data = data.float() # 图像数据标准化
            labels = labels.long()

            # 梯度清零
            optimizer.zero_grad()

            # 前向传播
            outputs = model(data)
            loss = criterion(outputs, labels)

            # 反向传播
            loss.backward()

            # torch.nn.utils.clip_grad_norm_(
            #     model.parameters(), 
            #     max_norm=10.0,  # 最大范数阈值
            #     norm_type=2    # L2范数
            # )
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            # 更新参数
            optimizer.step()
            scheduler.step()
            
        print(f"Epoch {epoch+1+round_single}/{num_epochs+rounds-1} {loss.item()}")


def test(device, model, test_loader):
    """
    针对 CIFAR-10 的测试函数
    CIFAR-10: 3通道, 32x32 彩色图像
    MNIST: 1通道, 28x28 灰度图像
    """
    model = model.to(device)
    model.eval()  # 切换到评估模式
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    accuracy = correct / total * 100
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy

def test_with_supcon(device, model, test_loader):
    """
    针对 CIFAR-10 的测试函数
    CIFAR-10: 3通道, 32x32 彩色图像
    MNIST: 1通道, 28x28 灰度图像
    """
    model = model.to(device)
    model.eval()  # 切换到评估模式
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            _, outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    accuracy = correct / total * 100
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy

if __name__ == "__main__":
    supcon = True
    print("开始训练...")
    log_str = str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))+"\n"
    train_time_1 = time.time()
    
    cuda_id = 0
    device = f"cuda:{cuda_id}"

    train_loader = get_train_loader()
    train_time_2 = time.time()
    print(f"加载训练集耗时: {train_time_2 - train_time_1}")
    log_str += f"加载训练集耗时: {train_time_2 - train_time_1}\n"
    print("总样本数:", len(train_loader.dataset))
    log_str += f"总样本数: {len(train_loader.dataset)}\n"
    if supcon:
        model = get_resnet18_supcon(projection_dim = -1)
        optimizer = optim.SGD(model.parameters(), lr=0.15)
        ce_criterion = nn.CrossEntropyLoss()
        supcon_criterion = SupConLoss(temperature=0.07)
        train_with_supcon(model, device, train_loader, optimizer, ce_criterion, supcon_criterion, num_epochs=5)
    else:
        model = get_resnet18()
        optimizer = optim.SGD(model.parameters(), lr=0.15)
        criterion = nn.CrossEntropyLoss()
        # 计算总迭代次数
        num_epochs = 5
        total_steps = num_epochs * len(train_loader)
        # 创建余弦学习率调度器
        # T_max是余弦周期的长度，eta_min是最小学习率
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
        train(device, model, train_loader, optimizer, scheduler, criterion, num_epochs=num_epochs)
    
    # optimizer = optim.AdamW(model.parameters(), lr=0.15, weight_decay=5e-4, betas=(0.9, 0.999))
    # print("个例样本数:", len(train_loaders_iid[0].dataset))
    
    train_time_3 = time.time()
    print(f"训练耗时: {train_time_3 - train_time_2}")
    log_str += f"训练耗时: {train_time_3 - train_time_2}\n"
    torch.save(model, 'CNN_supcon.pth')

    test_time_1 = time.time()
    test_loader = get_test_loader()
    test_time_2 = time.time()
    print(f"加载测试集耗时: {test_time_2 - test_time_1}")
    log_str += f"加载测试集耗时: {test_time_2 - test_time_1}\n"
    # 加载整个模型
    # model = torch.load('CNN_supcon.pth', map_location=device)
    test_time_3 = time.time()
    print(f"加载模型耗时: {test_time_3 - test_time_2}")
    log_str += f"加载模型耗时: {test_time_3 - test_time_2}\n"
    if supcon:
        acc = test_with_supcon(device, model, test_loader)
    else:
        acc = test(device, model, test_loader)
    test_time_4 = time.time()
    print(f"测试耗时: {test_time_4 - test_time_3}")
    log_str += f"测试耗时: {test_time_4 - test_time_3}\nTest Accuracy: {acc}%\n\n"
    with open(f"log_supcon.txt", "a") as f:
        f.write(log_str)
