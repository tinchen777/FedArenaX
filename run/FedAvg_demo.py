
from typing import (List, Any)

from fedarenax.general import Device
from fedarenax.client_utils import Client
from fedarenax.server_utils import Server
from fedarenax.model import SimpleNN, SimpleCNN, get_resnet18
from fedarenax.dataloader import get_train_loaders_iid, get_train_loaders_non_iid, get_test_loader


ROUND_NUM = 100
CLIENT_NUM = 10
CLIENT_SELECT_NUM = 4
SAVE_FOLDER = None


def FedAvg(data: Any, labels: Any):
    # TODO
    return ...


def select_clients(clients: List[Client], num_clients: int) -> List[Client]:
    # TODO
    return clients


if __name__ == "__main__":
    # init device
    Device.set_device_for_class("auto")
    # init global model
    GLOBAL_MODEL = ...
    # init server and  clients
    server = Server(global_model=GLOBAL_MODEL, save_folder=SAVE_FOLDER)
    CLIENTS: List[Client] = []
    for i in range(1, CLIENT_NUM+1):
        CLIENTS.append(Client(
            global_model=GLOBAL_MODEL,
            lr=0.01,
            lr_scheduler_pattern=f"cosine_lr-{ROUND_NUM}",
            momentum=0.9,
            user_id=i,
            save_folder=SAVE_FOLDER
        ))
    # init data loaders
    get_train_loaders_iid(CLIENTS)

    # start
    for round_id in range(1, ROUND_NUM+1):
        local_weights = []

        # broadcast global model
        for client in CLIENTS:
            client.weight_sync(server.model.state_dict())
            client.scheduler_sync(round_id)
        # clients local training
        for client in select_clients(CLIENTS, CLIENT_SELECT_NUM):
            client.step_training(
                round_id,
                train_algo=FedAvg,
                local_eps=1,
                grad_clip=None
            )
            local_weights.append(client.model.state_dict())
        # server aggregate local models
        server.aggregate.weights(
            pattern="Average",
            weights_list=local_weights,
            interaction="overwrite"
        )
        server.evaluation(
            round_id,
            eval_algo=FedAvg
        )
