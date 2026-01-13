import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

from src.fedarenax import get_resnet18, get_resnet18_fedproto, get_train_loader, get_test_loader, FedProtoLoss
import time

def train_with_fedproto(model, device, data_loader, optimizer, fedproto_criterion, global_protos=None, num_epochs=1, round_single=0, rounds=1):
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
            
            loss = fedproto_criterion(logits, labels, features, global_protos)

            # 反向传播
            loss.backward()
            
            # torch.nn.utils.clip_grad_norm_(
            #     model.parameters(), 
            #     max_norm=5.0,  # 最大范数阈值
            #     norm_type=2    # L2范数
            # )

            # 更新参数
            optimizer.step()
            
        print(f"Epoch {epoch+1+round_single}/{num_epochs+rounds-1} {loss.item()}")
        log_str += f"Epoch {epoch+1+round_single}/{num_epochs+rounds-1} {loss.item()}\n"
    return log_str
def test_with_fedproto(model, device, global_protos, test_loader):
    model.eval()
    test_loss = 0
    correct_head = 0  # 使用模型分类器头的准确率
    correct_proto = 0 # 使用原型距离计算的准确率
    total = 0

    # 确保全局原型已转移到正确的设备并转换为 tensor
    # global_protos: {label: tensor(512)}
    proto_labels = sorted(global_protos.keys())
    proto_tensor = torch.stack([global_protos[i] for i in proto_labels]).to(device)
    proto_labels = torch.tensor(proto_labels).to(device)

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            # 1. 模型前向传播
            features, logits = model(images)
            
            # 2. 计算交叉熵损失 (针对模型头)
            test_loss += F.cross_entropy(logits, labels, reduction='sum').item()
            
            # 3. 预测方式 A: 直接使用分类器输出 (Classifier Head)
            pred_head = logits.argmax(dim=1, keepdim=True)
            correct_head += pred_head.eq(labels.view_as(pred_head)).sum().item()
            
            # 4. 预测方式 B: 基于原型距离 (Nearest Prototype)
            # 计算特征与每个原型之间的欧氏距离
            # features: [batch, 512], proto_tensor: [num_classes, 512]
            dist = torch.cdist(features, proto_tensor, p=2) # 得到 [batch, num_classes]
            
            # 找到距离最近的原型索引
            closest_proto_idx = dist.argmin(dim=1)
            pred_proto = proto_labels[closest_proto_idx]
            correct_proto += pred_proto.eq(labels).sum().item()
            
            total += labels.size(0)

    accuracy_head = 100. * correct_head / total
    accuracy_proto = 100. * correct_proto / total
    return accuracy_head, accuracy_proto

def train(device, model, data_loader, optimizer, criterion, num_epochs=1, round_single=0, rounds=1):
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
            #     max_norm=5.0,  # 最大范数阈值
            #     norm_type=2    # L2范数
            # )

            # 更新参数
            optimizer.step()
            
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

if __name__ == "__main__":
    proto = True
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
    if proto:
        model = get_resnet18_fedproto()
        optimizer = optim.SGD(model.parameters(), lr=0.15)
        fedproto_criterion = FedProtoLoss(lamda=1.0)
        train_with_fedproto(model, device, train_loader, optimizer, fedproto_criterion, global_protos = None, num_epochs=5)
    else:
        model = get_resnet18()
        optimizer = optim.SGD(model.parameters(), lr=0.15)
        criterion = nn.CrossEntropyLoss()
        train(device, model, train_loader, optimizer, criterion, num_epochs=5)
    
    # optimizer = optim.AdamW(model.parameters(), lr=0.15, weight_decay=5e-4, betas=(0.9, 0.999))
    # print("个例样本数:", len(train_loaders_iid[0].dataset))
    
    train_time_3 = time.time()
    print(f"训练耗时: {train_time_3 - train_time_2}")
    log_str += f"训练耗时: {train_time_3 - train_time_2}\n"
    torch.save(model, 'CNN_fedproto.pth')

    test_time_1 = time.time()
    test_loader = get_test_loader()
    test_time_2 = time.time()
    print(f"加载测试集耗时: {test_time_2 - test_time_1}")
    log_str += f"加载测试集耗时: {test_time_2 - test_time_1}\n"
    # 加载整个模型
    # model = torch.load('CNN_fedproto.pth', map_location=device)
    test_time_3 = time.time()
    print(f"加载模型耗时: {test_time_3 - test_time_2}")
    log_str += f"加载模型耗时: {test_time_3 - test_time_2}\n"
    if proto:
        acc = test_with_fedproto(device, model, test_loader)
    else:
        acc = test(device, model, test_loader)
    test_time_4 = time.time()
    print(f"测试耗时: {test_time_4 - test_time_3}")
    log_str += f"测试耗时: {test_time_4 - test_time_3}\nTest Accuracy: {acc}%\n\n"
    with open(f"log_fedproto.txt", "a") as f:
        f.write(log_str)
