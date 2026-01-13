import torch
import torch.nn as nn
import torch.optim as optim
import copy
import time
import sys
import random

from src.fedarenax import get_resnet18_fedproto, get_test_loader, federated_averaging
from src.fedarenax import get_train_loaders_iid, get_train_loaders_non_iid, FedProtoLoss, get_local_prototypes, aggregate_global_prototypes
from proto import train_with_fedproto, test_with_fedproto

def set_seed(seed=2026, deterministic=True, benchmark=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(device, lamda, train_loaders, test_loader, client_num, rounds, epochs, learning_rate_start, learning_rate_end, other_txt, seed):
    set_seed(seed)
    init_model = get_resnet18_fedproto()
    # FedAvg
    global_protos = None
    optimizers = []
    local_models = []
    fedproto_criterions = []
    
    log_str = str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))+"\nseed:"+str(seed)+"\n"
    time_1 = time.time()

    for client in range(client_num):
        local_model = copy.deepcopy(init_model)
        local_models.append(local_model)
        optimizers.append(optim.SGD(local_model.parameters(), lr=learning_rate_start))
        fedproto_criterions.append(FedProtoLoss(lamda=lamda))
    for r in range(rounds):
        local_protos_list = []
        local_protos_count_list = []
        
        # selected_num = int(client_num * 0.3)
        # clients = list(range(client_num))
        # random.shuffle(clients)

        # # 选择前30%的客户端
        # for client in clients[:selected_num]:
        for client in range(client_num):
            learning_rate = learning_rate_start + (learning_rate_end - learning_rate_start) * r / rounds
            optimizers[client] = optim.SGD(local_models[client].parameters(), lr=learning_rate)
            print(f"Round:{r+1}/{rounds},Client:{client+1}")
            log_str_train = train_with_fedproto(local_models[client], device, train_loaders[client], optimizers[client], 
                              fedproto_criterions[client], global_protos, num_epochs=epochs, round_single=r, rounds=rounds)
            local_protos, local_protos_count = get_local_prototypes(local_models[client], train_loaders[client], device)
            local_protos_list.append(local_protos)
            local_protos_count_list.append(local_protos_count)
            log_str += f"[train.py]{log_str_train}\n"
        global_protos = aggregate_global_prototypes(local_protos_list, local_protos_count_list)

        acc_head, acc_protos = test_with_fedproto(local_models[0], device, global_protos, test_loader)
        time_2 = time.time()
        log_str += f"Round:{r+1}/{rounds},Acc_head:{acc_head:.4f},Acc_protos:{acc_protos:.4f},Time:{time_2-time_1:.2f}\n"
        time_1 = time_2
    # test_more(global_model, test_loader)
    with open(f"log_fedproto_{other_txt}_clientnum{client_num}_rounds{rounds}_epochs{epochs}_lr{learning_rate_start}:{learning_rate_end}.txt", "a") as f:
        f.write(log_str)

if __name__ == "__main__":
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
        main(device, 1.0, train_loaders_iid, test_loader, client_num, rounds, epochs, learning_rate_start,learning_rate_end, "iid", 2026+i)
        main(device, 1.0, train_loaders_non_iid, test_loader, client_num, rounds, epochs, learning_rate_start,learning_rate_end, "noniid", 2026+i)
# python FedProto.py 5 100 1 0.05 0.001 0