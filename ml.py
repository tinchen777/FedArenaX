import torch
import torch.nn as nn
import torch.optim as optim

from src.fedarenax import SimpleNN, train_loader, test_loader


model = SimpleNN()

optimizer = optim.SGD(model.parameters(), lr=0.01)

criterion = nn.CrossEntropyLoss()


def train(model, data_loader, optimizer, criterion, num_epochs=1):
    model.train()
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        for idx, (data, labels) in enumerate(data_loader, start=1):
            if idx % 100 == 0:
                print(f"  Batch {idx}")

            data = data.float().view(-1, 28*28) / 255.0  # 图像数据标准化
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


def test(model, test_loader):
    model.eval()  # 切换到评估模式
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in test_loader:
            outputs = model(data.view(-1, 28*28))  # 展平图像
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total * 100
    print(f"Test Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    # Machine learning

    train(model, train_loader, optimizer, criterion, num_epochs=10)

    test(model, test_loader)
