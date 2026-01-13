import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from src.fedarenax import trainset

def select_images_by_class(dataset, target_class=5, num_images=5):
    """选择特定类别的图片"""
    selected_indices = []
    selected_images = []
    
    for i in range(len(dataset)):
        image, label = dataset[i]
        if label == target_class:
            selected_indices.append(i)
            selected_images.append((image, label))
            if len(selected_images) >= num_images:
                break
    
    return selected_indices, selected_images
def prepare_mnist_image_for_cam(img_tensor):
    """
    准备MNIST图像用于Grad-CAM
    
    返回:
        image_tensor: 用于模型的张量 (1, 1, 28, 28)
        original_image: 用于可视化的原始图像 (28, 28)，值在0-1之间
    """
    # 1. 准备image_tensor（添加批次维度）
    image_tensor = img_tensor.unsqueeze(0)  # (1, 1, 28, 28)
    
    # 2. 准备original_image（反归一化）
    original_image = img_tensor.numpy().squeeze()  # (1, 28, 28) -> (28, 28)
    
    # 反归一化：x = (x_normalized * std) + mean
    original_image = (original_image * 0.5) + 0.5
    original_image = np.clip(original_image, 0, 1)  # 确保在0-1之间
    
    return image_tensor, original_image
def visualize_mnist_gradcam(model, image_tensor, original_image, target_class=None):
    """
    为MNIST图像生成Grad-CAM热力图
    
    参数:
        model: 训练好的模型
        image_tensor: 预处理后的图像张量 (1, 1, 28, 28)
        original_image: 原始图像数组 (28, 28)，值在0-1之间
        target_class: 目标类别，如果为None则使用模型预测的类别
    """
    model.eval()
    target_layers = [model.conv2]
    cam = GradCAM(
        model=model, 
        target_layers=target_layers
    )
    if target_class is None:
        with torch.no_grad():
            output = model(image_tensor)
            predicted_class = torch.argmax(output, dim=1).item()
        target_class = predicted_class
        print(f"模型预测的类别: {predicted_class}")
    
    # 4. 生成热力图
    # 注意：需要指定target_category
    grayscale_cam = cam(
        input_tensor=image_tensor
    )
    # 前向传播：将输入图像通过模型，得到模型输出（通常是softmax前的logits）。

    # 选择目标层：通常是卷积网络的最后一层卷积层，因为这一层保留了空间信息，且语义层次高。

    # 计算梯度：对于目标类别（通常是模型预测的类别），计算目标层每个通道的梯度。

    # 权重计算：对每个通道的梯度进行全局平均池化，得到每个通道的权重。

    # 加权组合：将目标层的每个通道乘以其对应的权重，然后求和，得到初步的激活图。

    # ReLU激活：通过ReLU函数，只保留对类别有正面影响的特征。

    # 上采样：将激活图上采样到输入图像的大小，得到热图。
    
    # 5. 将灰度图像转换为3通道用于可视化
    # MNIST是单通道，但show_cam_on_image需要3通道
    original_image_3ch = np.stack([original_image]*3, axis=-1)
    
    # 6. 将热力图叠加到原始图像
    visualization = show_cam_on_image(
        original_image_3ch, 
        grayscale_cam[0], 
        use_rgb=True
    )
    
    return grayscale_cam[0], visualization, target_class
def visualize_heatmap(selected_images, model, save_path):
    fig, axes = plt.subplots(len(selected_images), 4, figsize=(16, 4*len(selected_images)))

    for idx, (img_tensor, true_label) in enumerate(selected_images):
        # 准备图像
        image_tensor, original_image = prepare_mnist_image_for_cam(img_tensor)
        
        # 生成热力图
        heatmap, visualization, predicted_class = visualize_mnist_gradcam(
            model, image_tensor, original_image
        )
        
        # 显示结果
        # 1. 原始图像
        axes[idx, 0].imshow(original_image, cmap='gray')
        axes[idx, 0].set_title(f'True Label: {true_label}')
        axes[idx, 0].axis('off')
        
        # 2. 模型预测概率
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.softmax(output, dim=1)[0]
        
        axes[idx, 1].bar(range(10), probabilities.cpu().numpy())
        axes[idx, 1].set_title(f'Predicted: {predicted_class}\nConfidence: {probabilities[predicted_class]:.2%}')
        axes[idx, 1].set_xlabel('Digit')
        axes[idx, 1].set_ylabel('Probability')
        axes[idx, 1].set_ylim([0, 1])
        
        # 3. 热力图
        axes[idx, 2].imshow(heatmap, cmap='jet')
        axes[idx, 2].set_title('Grad-CAM Heatmap')
        axes[idx, 2].axis('off')
        
        # 4. 叠加可视化
        axes[idx, 3].imshow(visualization)
        axes[idx, 3].set_title('Overlay')
        axes[idx, 3].axis('off')

    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图像已保存至: {save_path}")

if __name__ == "__main__":
    model = torch.load('CNN.pth')
    target_class = 5
    indices, selected_images = select_images_by_class(trainset, target_class, num_images=3)
    visualize_heatmap(selected_images, model, "heatmap.svg")


