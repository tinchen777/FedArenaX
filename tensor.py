import torch

# 1. 创建一个可求导的 Tensor
x = torch.tensor([2.0, 3.0], requires_grad=True)
print("x:", x)

# 2. 定义一个简单函数 y = x1^2 + 2*x2^3
y = x[0]**2 + 2 * x[1]**3
print("y:", y)

# 3. 自动计算梯度
y.backward()  # 反向传播，计算 dy/dx

# 4. 查看梯度
print("dy/dx:", x.grad)

# 5. 手动梯度验证
# dy/dx1 = 2*x1 = 4.0
# dy/dx2 = 6*x2^2 = 6*9 = 54.0



x = torch.tensor([2.0, 3.0], requires_grad=True)
y = (x[0] + x[1])**2
y.backward(retain_graph=True)
print("First backward, dy/dx:", x.grad)


# 清零梯度
# x.grad.zero_()
y.backward()
print("After zeroing, dy/dx:", x.grad)

