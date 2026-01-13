
import torch
from torch import nn
import torch.nn.functional as F


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
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 卷积层
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # 2x2
        
        # 全连接
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # 28x28 -> 14x14 -> 7x7
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.pool(torch.relu(self.conv1(x)))  # 28x28 -> 14x14
        x = self.pool(torch.relu(self.conv2(x)))  # 14x14 -> 7x7
        
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
import torchvision.models as models

def get_resnet18(num_classes=10):
    """
    使用torchvision的预定义ResNet-18
    """
    model = models.resnet18()
    
    # # 修改最后一层以适应CIFAR-10（10类）
    # num_features = model.fc.in_features
    # model.fc = nn.Linear(num_features, num_classes)
    
    # 对于CIFAR-10（32x32图像），修改第一个卷积层
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model

class FeatureResNet18(nn.Module):
    """
    修改ResNet-18使其返回特征和分类输出
    """
    def __init__(self, num_classes=10):
        super(FeatureResNet18, self).__init__()
        
        # 加载预训练ResNet-18
        self.resnet = models.resnet18(pretrained=True)
        
        # 修改第一个卷积层适应CIFAR-10 (32x32图像)
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.maxpool = nn.Identity()
        
        # 保存原始的全连接层
        self.num_features = self.resnet.fc.in_features
        
        # 修改全连接层
        self.resnet.fc = nn.Linear(self.num_features, num_classes)
        
        # 将特征提取部分和分类部分分开
        self.feature_extractor = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.maxpool,
            self.resnet.layer1,
            self.resnet.layer2,
            self.resnet.layer3,
            self.resnet.layer4,
            self.resnet.avgpool,
            nn.Flatten()
        )
        
        self.classifier = self.resnet.fc
        
    def forward(self, x):
        """
        返回特征和分类输出
        Args:
            x: 输入图像
        Returns:
            features: 特征向量 [batch_size, num_features]
            outputs: 分类输出 [batch_size, num_classes]
        """
        # 提取特征
        features = self.feature_extractor(x)
        
        # 分类
        outputs = self.classifier(features)
        
        return features, outputs
    
def get_resnet18_supcon(num_classes=10, projection_dim = -2):
    """
    获取修改后的ResNet-18，可以返回特征和分类输出
    """
    # if projection_dim == -1:
    #     model = FeatureResNet18(num_classes=num_classes)
    # else:
    #     model = SupConResNet18(num_classes=num_classes, projection_dim=projection_dim)
    if projection_dim == -2:
        model = FeatureResNet18(num_classes=num_classes)
    else:
        model = SupConResNet18(num_classes=num_classes, projection_dim=projection_dim)
    return model

class SupConResNet18(nn.Module):
    """
    符合SupCon论文标准的ResNet-18
    论文: "Supervised Contrastive Learning" (NeurIPS 2020)
    
    特征流向：
    输入 → Backbone → 特征向量 → 分类头 → 分类输出
                        ↓
                    投影头 → 对比特征 → SupCon损失
    """
    def __init__(self, num_classes=10, projection_dim=128):
        """
        Args:
            num_classes: 分类任务类别数
            projection_dim: 投影头输出维度（论文中常用128）
            use_projection: 是否使用投影头（测试时可关闭）
        """
        super(SupConResNet18, self).__init__()
        
        # 1. Backbone: ResNet-18特征提取器
        self.backbone = models.resnet18(pretrained=True)  # 加载预训练权重
        
        # 修改适应CIFAR-10（32x32图像）
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.maxpool = nn.Identity()
        
        # 获取特征维度（ResNet-18最后一层是512维）
        self.feature_dim = 512  # ResNet-18的原始特征维度
        
        # 移除原始分类头
        self.backbone.fc = nn.Identity()
        
        # 2. 分类头（用于下游分类任务）
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        
        # 3. 投影头（用于对比学习，论文中使用两层MLP）
        if projection_dim == -1:
            self.use_projection = False
        else:
            self.use_projection = True
        if self.use_projection:
            self.projection_head = self._create_projection_head(projection_dim)
        
        # 初始化权重
        self._initialize_weights()
    
    def _create_projection_head(self, projection_dim=128):
        """
        创建投影头（符合论文中的两层MLP结构）
        论文中：特征 → 全连接层(ReLU) → 全连接层 → L2归一化
        """
        projection_head = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.feature_dim, projection_dim),
        )
        return projection_head
    
    def _initialize_weights(self):
        """初始化权重"""
        # 分类头初始化
        nn.init.normal_(self.classifier.weight, std=0.01)
        nn.init.constant_(self.classifier.bias, 0)
        
        # 投影头初始化（如果存在）
        if hasattr(self, 'projection_head'):
            for m in self.projection_head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.01)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入图像 [batch_size, 3, 32, 32]
            return_projection: 是否返回投影特征
            
        Returns:
            如果 return_projection=True: (features, projections, logits)
            否则: logits
        """
        # 1. 通过backbone提取特征
        features = self.backbone(x)  # [batch_size, 512]
        
        
        # 3. 投影特征（用于对比学习）
        if self.use_projection and hasattr(self, 'projection_head'):
            features = F.normalize(features, dim=1)  # L2归一化
            # 2. 分类输出
            logits = self.classifier(features)
            projections = self.projection_head(features)
            projections = F.normalize(projections, dim=1)  # 投影后L2归一化
            return projections, logits
        
        logits = self.classifier(features)
        return features, logits

def get_resnet18_fedproto(num_classes=10):
    model = FedProtoResNet18(num_classes=num_classes)
    
class FedProtoResNet18(nn.Module):
    """
    FedProto模型，基于ResNet-18
    训练时，返回特征向量和分类输出
    """
    def __init__(self, num_classes=10):
        super(FedProtoResNet18, self).__init__()
        # 加载标准 resnet18
        base_model = models.resnet18(pretrained=True) 
        
        # 修改第一层卷积以适配 32x32 图片
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        # 移除原有的 maxpool 层（CIFAR-10 图片太小，不需要第一层的池化）
        
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        self.avgpool = base_model.avgpool
        
        # 特征维度：ResNet18 最后一层输出是 512
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        # 特征提取
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        features = torch.flatten(x, 1) # 这里得到 512 维特征
        
        # 分类输出
        logits = self.classifier(features)
        return features, logits