import torch
import torch.nn as nn
import torch.optim as optim
import copy
import time
import sys
import random

from src.fedarenax import get_resnet18_supcon, get_test_loader, federated_averaging
from src.fedarenax import get_train_loaders_iid, get_train_loaders_non_iid, SupConLoss
from supcon import train_with_supcon, test_with_supcon

def set_seed(seed=2026, deterministic=True, benchmark=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(device, dim, train_loaders, test_loader, client_num, rounds, epochs, learning_rate_start, learning_rate_end, other_txt, seed):
    set_seed(seed)
    global_model = get_resnet18_supcon(projection_dim = dim)
    # FedAvg
    optimizers = []
    local_models = []
    ce_criterions = []
    supcon_criterions = []
    
    log_str = str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))+"\nseed:"+str(seed)+"\n"
    time_1 = time.time()

    for client in range(client_num):
        local_model = copy.deepcopy(global_model)
        local_model.to(device)
        local_models.append(local_model)
        optimizers.append(optim.SGD(local_model.parameters(), lr=learning_rate_start))
        ce_criterions.append(nn.CrossEntropyLoss())
        supcon_criterions.append(SupConLoss())
    global_model.to(device)
    for r in range(rounds):
        
        # selected_num = int(client_num * 0.3)
        # clients = list(range(client_num))
        # random.shuffle(clients)

        # # 选择前30%的客户端
        # for client in clients[:selected_num]:
        for client in range(client_num):
            local_models[client].load_state_dict(global_model.state_dict())
            learning_rate = learning_rate_start + (learning_rate_end - learning_rate_start) * r / rounds
            optimizers[client] = optim.SGD(local_models[client].parameters(), lr=learning_rate)
            print(f"Round:{r+1}/{rounds},Client:{client+1}")
            log_str_train = train_with_supcon(local_models[client], device, train_loaders[client], optimizers[client], 
                              ce_criterions[client], supcon_criterions[client], num_epochs=epochs, round_single=r, rounds=rounds)
            log_str += f"[train.py]{log_str_train}"
        global_dict = federated_averaging(global_model, local_models)
        global_model.load_state_dict(global_dict)

        acc = test_with_supcon(device, global_model, test_loader)
        time_2 = time.time()
        log_str += f"Round:{r+1}/{rounds},Acc:{acc:.4f},Time:{time_2-time_1:.2f}\n"
        time_1 = time_2
    # test_more(global_model, test_loader)
    with open(f"log_fed_supcon_{dim}_{other_txt}_clientnum{client_num}_rounds{rounds}_epochs{epochs}_lr{learning_rate_start}:{learning_rate_end}.txt", "a") as f:
        f.write(log_str)

if __name__ == "__main__":
    # args -- dim = 128
    # args = sys.argv[1:]
    # times = int(args[0])
    # CLIENT_NUM = 5
    # ROUNDS = 10
    # train_loaders_iid = get_train_loaders_iid(CLIENT_NUM)
    # test_loader = get_test_loader()
    # ALPHA = 0.5
    # train_loaders_non_iid = get_train_loaders_non_iid(CLIENT_NUM, ALPHA)
    # SEED = 2026
    # random.seed(SEED)
    # torch.manual_seed(SEED)
    # torch.cuda.manual_seed_all(SEED)
    # for i in range(times):
    #     print(f"Time:{i+1}/{times}")
    #     # main(-2)
    #     # main(-1)
    #     main(128, train_loaders_non_iid)
    # (device, dim, train_loaders, test_loader, client_num, rounds, epochs, learning_rate, other_txt, seed)
    args = sys.argv[1:]
    times = int(args[0])
    rounds = int(args[1])
    epochs = int(args[2])
    learning_rate_start = float(args[3])
    learning_rate_end = float(args[4])
    cuda_id = int(args[5])
    device = f"cuda:{cuda_id}"
    client_num = 10
    train_loaders_iid = get_train_loaders_iid(client_num)
    train_loaders_non_iid = get_train_loaders_non_iid(client_num, 0.5)
    test_loader = get_test_loader()
    for i in range(times):
        print(f"Time:{i+1}/{times}")
        main(device, 256, train_loaders_iid, test_loader, client_num, rounds, epochs, learning_rate_start,learning_rate_end, "iid", 2026+i)
        main(device, 256, train_loaders_non_iid, test_loader, client_num, rounds, epochs, learning_rate_start,learning_rate_end, "noniid", 2026+i)
# python FedAvg_supcon.py 5 100 1 0.05 0.001 0