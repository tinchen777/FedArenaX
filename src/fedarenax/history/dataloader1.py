from .data import trainset, testset, split_non_iid_data, split_iid_data
from torch.utils.data import DataLoader

train_loader = None
test_loader = None

def get_train_loader():
    global train_loader
    if train_loader is None:
        train_loader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=1)
    return train_loader
def get_test_loader():
    global test_loader
    if test_loader is None:
        test_loader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=1)
    return test_loader