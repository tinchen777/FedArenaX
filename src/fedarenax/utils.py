
import torch


# def federated_averaging(global_model, local_models):
#     with torch.no_grad():
#         # 对每一层的权重进行平均
#         for global_param, *local_params in zip(global_model.parameters(), *local_models):
#             global_param.data.copy_(torch.mean(torch.stack([param.data for param in local_params]), dim=0))
def federated_averaging(global_model, client_models, client_weights = None):
    global_dict = global_model.state_dict()
    
    # 初始化平均字典
    for key in global_dict.keys():
        global_dict[key] = torch.zeros_like(global_dict[key], dtype=torch.float32)
    
    if client_weights:
        total_weight = sum(client_weights)
        for i, client_model in enumerate(client_models):
            client_dict = client_model.state_dict()
            weight = client_weights[i] / total_weight
            
            for key in global_dict.keys():
                global_dict[key] += client_dict[key] * weight
    else:
        weight = 1.0/len(client_models)
        for i, client_model in enumerate(client_models):
            client_dict = client_model.state_dict()
            
            for key in global_dict.keys():
                global_dict[key] += client_dict[key] * weight

    return global_dict

def get_local_prototypes(model, dataloader, device):
    model.eval()
    local_protos_sum = {} # 存储特征向量之和
    local_protos_count = {} # 存储每个类别的样本数

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            features, _ = model(images)
            
            for i in range(len(labels)):
                label = labels[i].item()
                feature = features[i].detach().to(device)
                
                if label not in local_protos_sum:
                    local_protos_sum[label] = feature
                    local_protos_count[label] = 1
                else:
                    local_protos_sum[label] += feature
                    local_protos_count[label] += 1
    
    # 计算均值：类别 i 的原型 = 特征总和 / 样本数
    local_protos = {}
    for label in local_protos_sum.keys():
        local_protos[label] = local_protos_sum[label] / local_protos_count[label]
        
    # 返回该客户端的原型字典和每个类的样本量（用于服务端加权）
    return local_protos, local_protos_count

def aggregate_global_prototypes(all_client_protos, all_client_counts):
    """
    all_client_protos: 列表，每个元素是一个 client 的 local_protos 字典
    all_client_counts: 列表，每个元素是一个 client 的 local_protos_count 字典
    """
    global_protos = {}
    total_counts_per_class = {}

    # 遍历每个客户端
    for client_idx in range(len(all_client_protos)):
        c_protos = all_client_protos[client_idx]
        c_counts = all_client_counts[client_idx]

        for label, proto in c_protos.items():
            count = c_counts[label]
            
            if label not in global_protos:
                # 初始加权特征：特征向量 * 样本数
                global_protos[label] = proto * count
                total_counts_per_class[label] = count
            else:
                global_protos[label] += proto * count
                total_counts_per_class[label] += count

    # 归一化：除以所有客户端在该类上的样本总数
    for label in global_protos.keys():
        global_protos[label] = global_protos[label] / total_counts_per_class[label]
        
    return global_protos
