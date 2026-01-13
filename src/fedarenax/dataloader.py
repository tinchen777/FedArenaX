from .data import trainset, testset, split_non_iid_data, split_iid_data
from torch.utils.data import DataLoader

# 数据加载器（批量读取）
train_loaders_iid = {}
train_loaders_non_iid = {}
train_loader = None
test_loader = None

# FIXME
def get_train_loaders_iid(client_num):
    global train_loaders_iid
    if client_num not in train_loaders_iid:
        train_loaders_iid_single = [DataLoader(trainset_single, batch_size=64, shuffle=True, num_workers=4) for trainset_single in split_iid_data(trainset, client_num)]
        train_loaders_iid[client_num] = train_loaders_iid_single
    return train_loaders_iid[client_num]

def get_train_loaders_non_iid(client_num, alpha):
    global train_loaders_non_iid
    if client_num not in train_loaders_non_iid:
        train_loaders_non_iid_single = [DataLoader(trainset_single, batch_size=64, shuffle=True, num_workers=4) for trainset_single in split_non_iid_data(trainset, client_num, alpha=alpha)]
        train_loaders_non_iid[client_num] = train_loaders_non_iid_single
    return train_loaders_non_iid[client_num]

def delete_train_loaders_iid(client_num):
    global train_loaders_iid
    if client_num in train_loaders_iid:
        del train_loaders_iid[client_num]
def delete_train_loaders_non_iid(client_num):
    global train_loaders_non_iid
    if client_num in train_loaders_non_iid:
        del train_loaders_non_iid[client_num]
def get_train_loader():
    global train_loader
    if train_loader is None:
        train_loader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    return train_loader
def get_test_loader():
    global test_loader
    if test_loader is None:
        test_loader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
    return test_loader