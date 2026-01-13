import torch
import torch.nn as nn
import torch.optim as optim
import copy

from src.fedarenax import SimpleNN, SimpleCNN, test_loader, federated_averaging
from src.fedarenax import train_loaders_non_iid as train_loaders
from ml import train, test, test_more


global_model = SimpleNN()
# global_model = SimpleCNN()

CLIENT_NUM = 10
ROUNDS = 5

if __name__ == "__main__":
    # FedAvg
    optimizers = []
    criterions = []
    local_models = []

    for client in range(CLIENT_NUM):
        local_model = copy.deepcopy(global_model)
        local_models.append(local_model)
        optimizers.append(optim.SGD(local_model.parameters(), lr=0.01))
        criterions.append(nn.CrossEntropyLoss())

    for r in range(ROUNDS):
        for client in range(CLIENT_NUM):
            local_models[client].load_state_dict(global_model.state_dict())
            # optimizers[client] = optim.SGD(local_models[client].parameters(), lr=0.1)
            print(f"Round:{r+1}/{ROUNDS},Client:{client+1}")
            train(local_models[client], train_loaders[client], optimizers[client], criterions[client], num_epochs=1, round_single=r, rounds=ROUNDS)
        global_dict = federated_averaging(global_model, local_models)
        global_model.load_state_dict(global_dict)
        test(global_model, test_loader)
    test_more(global_model, test_loader)