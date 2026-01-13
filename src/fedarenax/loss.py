import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfDefLoss(nn.Module):
    def __init__(self, weight=None, reduction='mean', ignore_index=-100):
        super(SelfDefLoss, self).__init__()
        self.weight = weight
        self.reduction = reduction
        self.ignore_index = ignore_index
        
    def forward(self, input, target):
        """
        参数:
            input: (N, C) 或 (N, C, d1, d2, ...)，模型输出（未经过softmax）
            target: (N,) 或 (N, d1, d2, ...)，真实标签
        """
        # 手动实现交叉熵损失
        log_softmax = F.log_softmax(input, dim=1)
        
        # 如果 target 是类索引
        if target.dim() == 1 or (target.dim() == 2 and target.size(1) == 1):
            # 使用 nll_loss
            loss = F.nll_loss(
                log_softmax, 
                target, 
                weight=self.weight,
                reduction=self.reduction,
                ignore_index=self.ignore_index
            )
        else:
            # 如果 target 是 one-hot 编码
            loss = -(target * log_softmax).sum(dim=1)
            if self.reduction == 'mean':
                loss = loss.mean()
            elif self.reduction == 'sum':
                loss = loss.sum()
                
        return loss

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, features):
        """
        参数:
            features: (2N, d) 其中每个样本有正样本对
            [z_1, z_1^+, z_2, z_2^+, ...]
        """
        batch_size = features.shape[0]
        device = features.device
        
        # 创建标签：每个样本的正样本是它的配对样本
        labels = torch.arange(batch_size, device=device)
        
        # 计算相似度矩阵
        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # 创建掩码，排除对角线（自己与自己）
        mask = torch.eye(batch_size, device=device).bool()
        
        # 正样本相似度
        pos_sim = similarity_matrix[~mask].view(batch_size, -1)
        
        # 计算InfoNCE损失
        logits = similarity_matrix
        logits[mask] = float('-inf')  # 排除自己
        
        loss = F.cross_entropy(logits, labels)
        
        return loss
class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss
    基于论文：Supervised Contrastive Learning (Khosla et al., 2020)
    """
    
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        """
        参数:
            temperature: 温度参数τ，控制相似度分布的平滑程度
            contrast_mode: 'all'表示使用所有样本对比，'one'表示只与正样本对比
            base_temperature: 基础温度参数
        """
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
    
    def forward(self, features, labels=None, mask=None):
        """
        前向传播
        
        参数:
            features: 特征向量 [batch_size, feature_dim] 或 [batch_size * n_views, feature_dim]
            labels: 标签 [batch_size] (如果是多视图，则标签会自动展开)
            mask: 自定义的对比掩码，如果不为None则忽略labels参数
            
        返回:
            loss: 对比损失
        """
        device = features.device
        
        # 获取批次大小和特征维度
        if len(features.shape) != 2:
            raise ValueError('`features` 需要是形状为 [batch_size, feature_dim] 的张量')
        
        batch_size = features.shape[0]
        
        # 如果没有提供mask，则使用labels构建mask
        if mask is not None:
            mask = mask.float().to(device)
        else:
            if labels is None:
                raise ValueError('`labels` 不能为None，除非提供了`mask`')
            
            # 确保labels是1D张量
            labels = labels.contiguous().view(-1, 1)
            
            # 构建mask：相同标签的位置为1，不同标签的位置为0
            if labels.shape[0] != batch_size:
                raise ValueError('`labels` 的批次大小与`features`不匹配')
            
            mask = torch.eq(labels, labels.T).float().to(device)
        
        # 计算所有样本间的相似度
        contrast_count = features.shape[0]
        contrast_feature = features
        
        # 特征归一化（L2归一化）
        contrast_feature = F.normalize(contrast_feature, p=2, dim=1)
        
        # 计算相似度矩阵
        anchor_dot_contrast = torch.div(
            torch.matmul(contrast_feature, contrast_feature.T),
            self.temperature
        )
        
        # 为了数值稳定性，减去最大值
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        # 创建对角线的mask（排除自身）
        logits_mask = torch.ones_like(mask)
        # 创建对角线为0的mask（排除自身）
        if self.contrast_mode == 'one':
            logits_mask = mask
        elif self.contrast_mode == 'all':
            logits_mask = torch.scatter(
                torch.ones_like(mask),
                1,
                torch.arange(batch_size).view(-1, 1).to(device),
                0
            )
            logits_mask = logits_mask * mask
        else:
            raise ValueError(f'Unknown contrast mode: {self.contrast_mode}')
        
        # 计算exp(logits)
        exp_logits = torch.exp(logits) * logits_mask
        
        # 计算log概率
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
        
        # 计算每个正样本对的平均log概率
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)
        
        # 计算损失
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()
        
        return loss
    
class FedProtoLoss(nn.Module):
    def __init__(self, lamda=1.0):
        super(FedProtoLoss, self).__init__()
        self.lamda = lamda
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

    def forward(self, logits, labels, features, global_protos):
        """
        logits: 模型输出 (batch_size, num_classes)
        labels: 真实标签 (batch_size)
        features: 提取的特征 (batch_size, 512)
        global_protos: 字典，包含全局原型 {class_id: prototype_tensor}
        """
        # 1. 计算基本的交叉熵损失
        loss_ce = self.ce_loss(logits, labels)
        
        # 2. 计算原型损失 (如果全局原型尚未建立，则只返回 CE)
        if global_protos is None or len(global_protos) == 0:
            return loss_ce
        
        # 提取当前 batch 样本对应的全局原型
        # 注意：global_protos 应该在 device 上
        batch_global_protos = []
        for label in labels:
            # 如果某个类别在全局原型中不存在，则用当前特征代替（即该样本不产生 proto 误差）
            proto = global_protos.get(label.item(), features[0].detach())
            batch_global_protos.append(proto)
        
        batch_global_protos = torch.stack(batch_global_protos).to(features.device)
        
        # 计算特征与对应全局原型的 MSE 距离
        loss_proto = self.mse_loss(features, batch_global_protos)
        
        return loss_ce + self.lamda * loss_proto