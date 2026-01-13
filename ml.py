import torch
import torch.nn as nn
import torch.optim as optim

from src.fedarenax import SimpleNN, SimpleCNN, train_loader, test_loader, transform


# model = SimpleNN()
model = SimpleCNN()

optimizer = optim.SGD(model.parameters(), lr=0.01)

criterion = nn.CrossEntropyLoss()

cuda_id = 0
device = f"cuda:{cuda_id}"

def train(model, data_loader, optimizer, criterion, num_epochs=1, round_single=0, rounds=1):
    model = model.to(device)
    model.train()
    for epoch in range(num_epochs):
        for idx, (data, labels) in enumerate(data_loader, start=1):
            data, labels = data.to(device), labels.to(device)
            if idx % 100 == 0:
                print(f"  Batch {idx}")

            # data = data.float().view(-1, 28*28) / 255.0  # 图像数据标准化
            data = data.float() # 图像数据标准化
            labels = labels.long()

            # 梯度清零
            optimizer.zero_grad()

            # 前向传播
            outputs = model(data)
            loss = criterion(outputs, labels)

            # 反向传播
            loss.backward()

            # 更新参数
            optimizer.step()
        print(f"Epoch {epoch+1+round_single}/{num_epochs+rounds-1} {loss.item()}")


def test(model, test_loader):
    model = model.to(device)
    model.eval()  # 切换到评估模式
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data.view(-1, 28*28))  # 展平图像
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total * 100
    print(f"Test Accuracy: {accuracy:.2f}%")

def test_more(model, test_loader):
    model.eval()  # 切换到评估模式
    correct = 0
    total = 0

    with torch.no_grad():
        sample_data, _ = next(iter(test_loader))
        sample_output = model(sample_data.view(-1, 28*28))
        num_classes = sample_output.size(1) 

    confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)
    
    with torch.no_grad():
        for data, labels in test_loader:
            outputs = model(data.view(-1, 28*28))  # 展平图像
            _, predicted = torch.max(outputs, 1)
            
            # 更新混淆矩阵
            for t, p in zip(labels.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    accuracy = correct / total * 100
    print(f"Test Accuracy: {accuracy:.2f}%")
    
    # 计算各种指标
    print("\n详细指标:")
    print("-" * 50)
    
    # 计算每个类别的指标
    for i in range(num_classes):
        # 真阳性 (TP): 正确预测为该类别的样本
        TP = confusion_matrix[i, i]
        
        # 假阳性 (FP): 错误预测为该类别的其他类别样本
        FP = confusion_matrix[:, i].sum() - TP
        
        # 假阴性 (FN): 属于该类但被错误预测为其他类别的样本
        FN = confusion_matrix[i, :].sum() - TP
        
        # 真阴性 (TN): 正确预测为其他类别的样本
        TN = confusion_matrix.sum() - (TP + FP + FN)
        
        # 计算精确率、召回率和F1分数
        precision = TP.float() / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP.float() / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"类别 {i}:")
        print(f"  真阳性 (TP): {TP.item()}, 假阳性 (FP): {FP.item()}, 假阴性 (FN): {FN.item()}")
        print(f"  精确率: {precision:.4f}, 召回率: {recall:.4f}, F1分数: {f1:.4f}")
    
    # 计算总体指标
    total_TP = torch.diag(confusion_matrix).sum().float()
    total_FP = confusion_matrix.sum(dim=0).float().sum() - total_TP
    total_FN = confusion_matrix.sum(dim=1).float().sum() - total_TP
    
    macro_precision = torch.diag(confusion_matrix).float() / (confusion_matrix.sum(dim=0).float() + 1e-10)
    macro_recall = torch.diag(confusion_matrix).float() / (confusion_matrix.sum(dim=1).float() + 1e-10)
    
    print("\n总体指标:")
    print(f"总真阳性: {total_TP.item()}")
    print(f"总假阳性: {total_FP.item()}")
    print(f"总假阴性: {total_FN.item()}")
    print(f"宏平均精确率: {macro_precision.mean():.4f}")
    print(f"宏平均召回率: {macro_recall.mean():.4f}")
    
    return confusion_matrix, accuracy

if __name__ == "__main__":
    # Machine learning
    print("总样本数:", len(train_loader.dataset))
    # print("个例样本数:", len(train_loaders_iid[0].dataset))
    train(model, train_loader, optimizer, criterion, num_epochs=5)
    torch.save(model, 'CNN.pth')

    # 加载整个模型
    loaded_model = torch.load('CNN.pth')
    test(model, test_loader)
