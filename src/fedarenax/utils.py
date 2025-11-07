
import torch


def federated_averaging(global_model, local_models):
    with torch.no_grad():
        # 对每一层的权重进行平均
        for global_param, *local_params in zip(global_model.parameters(), *local_models):
            global_param.data.copy_(torch.mean(torch.stack([param.data for param in local_params]), dim=0))
