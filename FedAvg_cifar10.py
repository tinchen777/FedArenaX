import torch
import torch.nn as nn
import torch.optim as optim
import copy
import time

from src.fedarenax import get_resnet18, get_test_loader, federated_averaging
from src.fedarenax import get_train_loaders_iid, get_train_loaders_non_iid
from supcon import train, test


CLIENT_NUM = 5
ROUNDS = 10

global_model = get_resnet18()
train_loaders = get_train_loaders_iid(CLIENT_NUM)
test_loader = get_test_loader()

if __name__ == "__main__":
    cuda_id = 0
    device = f"cuda:{cuda_id}"
    # FedAvg
    optimizers = []
    criterions = []
    local_models = []
    
    log_str = str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))+"\n"
    time_1 = time.time()

    for client in range(CLIENT_NUM):
        local_model = copy.deepcopy(global_model)
        local_models.append(local_model)
        optimizers.append(optim.SGD(local_model.parameters(), lr=0.3))
        criterions.append(nn.CrossEntropyLoss())
    global_model.to(device)
    for r in range(ROUNDS):
        for client in range(CLIENT_NUM):
            local_models[client].load_state_dict(global_model.state_dict())
            optimizers[client] = optim.SGD(local_models[client].parameters(), lr=0.15)
            print(f"Round:{r+1}/{ROUNDS},Client:{client+1}")
            train(device, local_models[client], train_loaders[client], optimizers[client], criterions[client], num_epochs=1, round_single=r, rounds=ROUNDS)
        global_dict = federated_averaging(global_model, local_models)
        global_model.load_state_dict(global_dict)

        acc = test(device, global_model, test_loader)
        time_2 = time.time()
        log_str += f"Round:{r+1}/{ROUNDS},Acc:{acc:.4f},Time:{time_2-time_1:.2f}\n"
        time_1 = time_2
    # test_more(global_model, test_loader)
    with open("log_fed_cifar10.txt", "a") as f:
        f.write(log_str)