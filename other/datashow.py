import torch
import torch.nn as nn
import torch.optim as optim
import copy

from src.fedarenax import trainset, split_non_iid_data, split_iid_data, visualize_client_distribution

CLIENT_NUM = 10

client_datasets_iid = split_iid_data(trainset, CLIENT_NUM)
client_datasets_noniid = split_non_iid_data(trainset, CLIENT_NUM, alpha=0.5)

# 打印统计信息
# print_statistics(client_datasets_iid, trainset)
visualize_client_distribution(client_datasets_iid, trainset, save_path="./iid.svg")
visualize_client_distribution(client_datasets_noniid, trainset, save_path="./noniid.svg")
