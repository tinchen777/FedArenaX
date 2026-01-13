
import torch
from torch import nn
from torch.utils.data import DataLoader
import copy
from tqdm import tqdm
import logging
from typing import (Any, Optional, Callable, OrderedDict)

from general import Device


class Client(Device):
    def __init__(
        self,
        global_model: nn.Module,
        lr: float = 0.1,
        lr_scheduler_pattern: Optional[str] = None,
        momentum: float = 0.1,
        user_id: int = 1,
        save_folder: Optional[str] = None
    ):
        self.__USER_ID = user_id
        self._SAVE_FOLDER = save_folder
        # local model
        self.model = copy.deepcopy(global_model).to(self.DEVICE)
        # optimization method
        self.__optimizer = torch.optim.SGD(
            params=self.model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=5e-4  # lambda in L2 norm
        )
        # LR scheduler
        self.__lr_scheduler = LearningRateScheduler(
            optimizer=self.__optimizer,
            pattern=lr_scheduler_pattern
        )

    def train_loader_recv(self, user_trainloader: DataLoader[Any]):
        r"""
        Info
        ----
            Receive train dataset loader for user training.

        Params
        ----
            user_trainloader (_Optional[DataLoader[Any]]_): Train dateset loader for user.
        """
        try:
            # user train data loader
            assert isinstance(user_trainloader, DataLoader), "Invalid Train Dataset Loader"
            self.__USER_TRAINLOADER = user_trainloader

        except Exception as e:
            logging.error(f"{self.sign} [train_loader_recv()] Not Implemented", e)

    def step_training(self, cur_round: int, train_algo: Callable[..., Any], verbose: bool = True, local_eps: int = 1, grad_clip: Optional[float] = 10.0, **algo_kwargs: Any):
        r"""
        Info
        ----
            Execute a step of local training on train dataset for this user.

        Raises
        ----
            Any exception.
            NOTE: When `AttributeError` is to be raised, raise `MyException` instead.
        """
        logging.info(f"Start Train - Round[{cur_round}/]-{self.sign}")

        try:
            # set state
            self.model.train()

            # === start training ===
            for local_ep in range(1, local_eps+1):

                # == for each epoch ==

                # iterator
                if verbose:
                    if local_ep == local_eps:
                        pbar_leave = True
                    else:
                        pbar_leave = False
                    iterator = tqdm(
                        enumerate(self.train_loader, start=1),
                        total=len(self.train_loader),
                        leave=pbar_leave,
                        colour="green",
                        unit="batch(es)"
                    )
                else:
                    iterator = enumerate(self.train_loader, start=1)

                for batch_idx, (data, labels) in iterator:

                    # = for each batch =
                    logging.info(f"Round[{cur_round}/]-{self.sign}Train-Epoch[{local_ep}/{local_eps}]-Batch[{batch_idx}/{len(self.train_loader)}]")

                    # forward
                    try:
                        outputs, loss, labels = train_algo(
                            data=data.to(self.DEVICE),  # shape [bsz, 1, [img_matrix]]
                            labels=labels.to(self.DEVICE),  # shape [bsz]
                            **algo_kwargs
                        )
                    except RuntimeError:
                        logging.error(f"{self.sign} Runtime Error in Forward, Skip This Batch")
                        continue

                    if loss is None:
                        logging.error("Loss Is None, Backward Ignored")
                        continue

                    # backward
                    self.__optimizer.zero_grad()
                    loss.backward()
                    # clip gradients
                    if grad_clip is not None:
                        torch.nn.utils.clip_grad.clip_grad_norm_(
                            parameters=self.model.parameters(),
                            max_norm=grad_clip,
                            norm_type=2
                        )

                    self.__optimizer.step()

                    # calculate metrics
                    # TODO
                    
                    # progress bar
                    if verbose and isinstance(iterator, tqdm):
                        # FIXME
                        pass
                        # iterator.set_description(self._state.prefix_format_str)
                        # iterator.set_postfix_str(self._state.postfix_format_str)
                    # save for train result
                    # TODO
                    self._SAVE_FOLDER

                    # = end each batch =

                # update total epochs for user training
                self.__total_eps += 1
                # == end each epoch ==

            # === end training ===

        except KeyboardInterrupt:
            raise KeyboardInterrupt()
        except AttributeError as e:
            logging.error(f"{self.sign} Local Training Error", e)

    def weight_sync(self, new_weight: OrderedDict[str, Any]):
        r"""
        Info
        ----
            Synchronization for local model weight.

        Params
        ----
            new_weight (_OrderedDict[str, Any]_): New model weight, from `model.state_dict()`.
        """
        try:
            assert new_weight, "No New Weight"
            self.model.load_state_dict(new_weight)

        except Exception as e:
            logging.error(f"{self.sign} Local Model Weight Not Synchronization", e)

    def scheduler_sync(self, cur_round: int):
        r"""
        Info
        ----
            Synchronization for user scheduler of learning rate to current round step.
            To replace `scheduler.step()` in each local training step.

        Return
        ----
            float: LR
        """
        if self.__lr_scheduler:
            self.__lr_scheduler.sync_step(cur_round=cur_round)
        else:
            logging.error(f"{self.sign} Training Learning Rate Not Synchronization, No Scheduler For Learning Rate")

        return self.lr

    @property
    def sign(self):
        r"""
        Sign of this user, like "User[1]".
        """
        return f"User[{self.__USER_ID}]"

    @property
    def train_loader(self):
        r"""
        Current train dataset loader for this user's local training. Raise `AttributeError` if not exist.
        """
        try:
            return self.__USER_TRAINLOADER
        except Exception:
            raise AttributeError(f"{self.sign} Have No Train Dataset Loader")

    @property
    def lr(self):
        r"""
        Current learning rate for this user's local training. Raise `AttributeError` if `scheduler of learning rate` is `None`.
        """
        if self.__lr_scheduler is not None:
            return self.__lr_scheduler.lr
        else:
            raise AttributeError(f"{self.sign} Have No Train Learning Rate")


class LearningRateScheduler:
    r"""
    Class for scheduler of train learning rate, by control optimizer instance.
    """
    def __init__(self, optimizer: torch.optim.Optimizer, pattern: Optional[str]):
        r"""
        Info
        ----
            Class for scheduler of train learning rate.

        Params
        ----
            optimizer (_torch.optim.Optimizer_): Optimizer for training.

            pattern (_str_): Pattern for a scheduler.
                - LR衰减方式：None, exp_lr-0.9, step_lr-0.1, multi_step_lr-0.9, cosine_lr-200, lambda
                - 后续参数(LR_Decay|T_max)
                Raise `ValueError` if `pattern` is not matched.

        Raises
        ----
            `ValueError`: `pattern` not matched.
        """
        self.__optimizer = optimizer
        self.__scheduler = self.__init_scheduler(pattern=pattern)

        self.__last_round: int = 1  # The last round for learning rate Synchronization

    def __init_scheduler(self, pattern: Optional[str]):
        r"""
        Info
        ----
            Initialize scheduler.

        Return
        ----
            Scheduler instance

        Raises
        ----
            `ValueError`: `pattern` not matched.
        """
        schedulers = torch.optim.lr_scheduler
        instance = None

        if pattern:
            try:
                pattern_list = pattern.split("-")
                if pattern_list[0] == "exp_lr":
                    # ExpLR 指数衰减
                    instance = schedulers.ExponentialLR(
                        optimizer=self.__optimizer,
                        gamma=float(pattern_list[1]),
                        last_epoch=-1
                    )
                elif pattern_list[0] == "step_lr":
                    # StepLR 固定步长衰减
                    instance = schedulers.StepLR(
                        optimizer=self.__optimizer,
                        step_size=2,
                        gamma=float(pattern_list[1]),
                        last_epoch=-1
                    )
                elif pattern_list[0] == "multi_step_lr":
                    # MultiStepLR 多步长衰减
                    instance = schedulers.MultiStepLR(
                        optimizer=self.__optimizer,
                        milestones=[50, 100, 150],
                        gamma=float(pattern_list[1]),
                        last_epoch=-1
                    )
                elif pattern_list[0] == "cosine_lr":
                    # CosineLR 余弦退火衰减
                    instance = schedulers.CosineAnnealingLR(
                        optimizer=self.__optimizer,
                        T_max=int(pattern_list[1]),
                        eta_min=0.00001,
                        last_epoch=-1
                    )
                elif pattern_list[0] == "lambda":
                    # 自定义函数 1
                    instance = schedulers.LambdaLR(
                        optimizer=self.__optimizer,
                        lr_lambda=self.__my_scheduler,
                        last_epoch=-1
                    )
                elif pattern_list[0] == "lambda2":
                    # 自定义函数 2
                    instance = schedulers.LambdaLR(
                        optimizer=self.__optimizer,
                        lr_lambda=self.__my_scheduler2,
                        last_epoch=-1
                    )
                else:
                    raise ValueError(f"Can Not Found Pattern [{pattern}] For Scheduler")

            except Exception as e:
                error("No Scheduler Implemented", e, stack_info=True)
                raise ValueError()

        return instance

    @staticmethod
    def __my_scheduler(epoch):
        r"""
        Customized scheduler `1` of learning rate. Return factor of learning rate.
        """
        if epoch <= 50:
            factor = 1
        elif epoch <= 70:
            factor = 0.7
        elif epoch <= 100:
            factor = 0.5
        elif epoch <= 120:
            factor = 0.2
        else:
            factor = 0.1

        return factor

    @staticmethod
    def __my_scheduler2(epoch):
        r"""
        Customized scheduler `2` of learning rate. Return factor of learning rate.
        """
        return (2.0 - 0.1 * epoch)

    def sync_step(self, cur_round: int):
        r"""
        Info
        ----
            Synchronization for the learning rate at the given round by optimizer instance.

        Params
        ----
            cur_round (_int_): Given round.
        """
        step = cur_round - self.__last_round

        if step > 0 and self.__scheduler is not None:
            optimizer = copy.deepcopy(self.__optimizer)

            for _ in range(step):
                self.__optimizer.zero_grad()
                self.__optimizer.step()
                self.__scheduler.step()

            optimizer.param_groups[0]["lr"] = self.lr
            self.__optimizer = optimizer

            self.__last_round = cur_round

    @property
    def lr(self):
        r"""
        Current learning rate for training.
        """
        if self.__scheduler:
            lr = self.__scheduler.get_last_lr()[0]
        else:
            lr = float(self.__optimizer.param_groups[0]["lr"])

        return lr
