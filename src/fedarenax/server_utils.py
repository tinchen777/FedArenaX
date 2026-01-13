
import torch
from torch import nn
from torch.utils.data import DataLoader
import copy
from tqdm import tqdm
import logging
from typing import (Any, Optional, Callable, OrderedDict, Literal)

from general import Device
from . import myath as mth


class Server(Device):
    def __init__(
        self,
        global_model: nn.Module,
        save_folder: Optional[str] = None
    ):
        self._SAVE_FOLDER = save_folder
        # local model
        self.model = global_model.to(self.DEVICE)
        # Aggregation
        self.aggregate = Aggregation(self)

    def test_loader_recv(self, test_loader: DataLoader[Any]):
        r"""
        Info
        ----
            Receive test dataset loader for evaluation.

        Params
        ----
            test_loader (_Optional[DataLoader[Any]]_): Test dateset loader for user.
        """
        try:
            # test data loader
            assert isinstance(test_loader, DataLoader), "Invalid Test Dataset Loader"
            self.__TEST_LOADER = test_loader

        except Exception as e:
            logging.error(f"{self.sign} [test_loader_recv()] Not Implemented", e)

    def evaluation(self, cur_round: int, eval_algo: Callable[..., Any], verbose: bool = True, **algo_kwargs: Any):
        r"""
        Info
        ----
            Execute an evaluation.
            NOTE: A single forward propagation on test dataset.

        Raises
        ----
            Any exception except `AttributeError`.
        """
        logging.info(f"Start Test - Round[{cur_round}/]-{self.sign}")

        try:
            # set state
            self.model.eval()

            # === start evaluation ===

            # iterator
            if verbose:
                iterator = tqdm(
                    enumerate(self.test_loader, start=1),
                    total=len(self.test_loader),
                    leave=True,
                    colour="blue",
                    unit="batch(es)"
                )
            else:
                iterator = enumerate(self.test_loader, start=1)

            with torch.no_grad():
                for batch_idx, (data, labels) in iterator:

                    # == for each batch ==
                    logging.info(f"Round[{cur_round}/]-{self.sign}Test-Batch[{batch_idx}/{len(self.test_loader)}]")

                    # forward
                    outputs, loss, labels = eval_algo(
                        data=data.to(self.DEVICE),  # shape [bsz, 1, [img_matrix]]
                        labels=labels.to(self.DEVICE),  # shape [bsz]
                        **algo_kwargs
                    )

                    # calculate metrics
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

                    # == end each batch ==

            # === end evaluation ===

        except KeyboardInterrupt:
            raise KeyboardInterrupt()
        except AttributeError as e:
            logging.error(f"{self.sign} {repr(self)} Evaluation Error", e)

    def weight_sync(self, new_weight: OrderedDict[str, torch.Tensor], as_increment: bool = False):
        r"""
        Info
        ----
            Synchronization for global model weight.

        Params
        ----
            new_weight (_OrderedDict[str, torch.Tensor]_): New model weight, from `model.state_dict()`.

            as_increment (_bool_): Control whether to sync weights by `current global model weight` + `new weight`. NOTE: layers based on current model.
        """
        try:
            assert new_weight, "No New Weight"

            # sync as weight increment
            # if as_increment:
            #     try:
            #         new_weight = OrderedDict(mth.tensor_dict_cal(self.result.model_weight, new_weight, operator="add"))
            #     except Exception as e:
            #         raise KeyError("Add New Weight Increment Error", e)

            self.model.load_state_dict(new_weight)

        except Exception as e:
            logging.error(f"{self.sign} Global Model Weight Not Synchronization", e)

    @property
    def sign(self):
        r"""
        Sign of server, "Server"
        """
        return "Server"

    @property
    def test_loader(self):
        r"""
        Current test dataset loader. Raise `AttributeError` if not exist.
        """
        try:
            return self.__TEST_LOADER
        except Exception:
            raise AttributeError(f"{self.sign} Have No Test Dataset Loader")


class Aggregation(object):
    r"""
    Class of Aggregation.
    """
    def __init__(self, server: Server):
        r"""
        Info
        ----
            Initialze `Aggregation`.

        Params
        ----
            server (_Server_): A instance of `Server`.

        Raises
        ----
            `ValueError`.
        """
        if isinstance(server, Server):
            self.__server = server
        else:
            raise ValueError("[Aggregation] Initialize Error, Invalid Server Instance")

    def weights(self, pattern: Literal["Weighted", "Average"], weights_list: list[OrderedDict[str, torch.Tensor]], factors_list: Optional[list[Any]] = None, interaction: Optional[Literal["as_increment", "overwrite"]] = "overwrite"):
        r"""
        Info
        ----
            Aggregation of model weights from clients.

        Params
        ----
            weights_list (_list[OrderedDict[str, Any]]_): A list of model weights.

            factors_list (_Optional[list[Any]]_) OPTIONAL. A list of factors for weighted aggregation.

            interaction (_str_): The interaction to the current weight. `"as_increment"`: to aggregate by `current global model weight` + `weight increments`; `"overwrite"`: overwrite the current weight; `None`: No interaction.

        Return
        ----
            OrderedDict[str, torch.Tensor]: Aggregrated weight.
        """
        logging.info("Aggregation For Weight")

        num_weights = len(weights_list)

        if num_weights == 1:
            # for single client
            global_weight: OrderedDict[str, torch.Tensor] = copy.deepcopy(weights_list[0])

        else:
            # for multiple clients
            try:
                if pattern == "Weighted":

                    assert factors_list is not None, "No Weighted Aggregation Factors List"
                    assert len(factors_list) == num_weights, "Weights List And Weighted Aggregation Factors List Have Different Amount"

                    global_weight = OrderedDict(mth.tensor_dict_cal(*weights_list, operator="wt_avg", weights=factors_list))

                elif pattern == "Average":

                    global_weight = OrderedDict(mth.tensor_dict_cal(*weights_list, operator="avg"))

                else:
                    raise ValueError(f"Unsupported Aggregation Algorithm [{pattern}]")

            except Exception as e:
                logging.error(f"{self.__server.sign} Weight Aggregation Error", e)
                raise ValueError

        # interaction global weight
        if interaction:
            self.__server.weight_sync(global_weight, as_increment=interaction == "as_increment")

        return global_weight

    # def class_prototypes(self, class_prototypes_list: list[torch.Tensor], labels_counts_list: Optional[list[torch.Tensor]] = None):
    #     r"""
    #     Info
    #     ----
    #         Aggregation of class prototypes from clients.

    #     Params
    #     ----
    #         class_prototypes_list (_list[torch.Tensor]_): A list of class_prototypes, class_prototypes shape as `[num_classes, feature_dim]`.

    #         labels_counts_list (_list[torch.Tensor]_): OPTIONAL. A list of labels_counts corresponds to class_prototypes, labels_counts shape as `[num_classes]`.
    #     """
    #     info("Aggregation For Class Prototypes", throw=False)

    #     try:
    #         aggregation_algorithm = wg.dict_str(self.__server._ALGO_CONFIG, "class_prototypes_aggregation_algorithm")

    #         if aggregation_algorithm == "Weighted":

    #             assert labels_counts_list is not None, "No Labels Counts List"
    #             assert len(class_prototypes_list) == len(labels_counts_list), "Class Prototypes List And Labels Counts List Have Different Amount"

    #             class_prototypes_stack = torch.stack(class_prototypes_list)  # shape as [groups, num_classes, feature_dim]
    #             labels_counts_stack_expand = torch.stack(labels_counts_list).unsqueeze_(2).expand_as(class_prototypes_stack)  # shape as [groups, num_classes, feature_dim]

    #             weighted_class_prototypes_stack = class_prototypes_stack * labels_counts_stack_expand  # shape as [groups, num_classes, feature_dim]

    #             class_prototypes = weighted_class_prototypes_stack.nansum(0) / labels_counts_stack_expand.sum(0)

    #         elif aggregation_algorithm == "Average":

    #             class_prototypes = torch.stack(class_prototypes_list).nanmean(dim=0)

    #         else:
    #             raise ValueError(f"Unsupported Class Prototypes Aggregation Algorithm [{aggregation_algorithm}]")

    #     except Exception as e:
    #         critical(f"{self.__server.sign} Class Prototypes Aggregation Error", e)
    #     else:
    #         self.__server.class_prototypes_sync(class_prototypes)

    # def class_prototypes_from_tensors_banks(self, features_banks_list: list[torch.Tensor], labels_banks_list: list[torch.Tensor]):
    #     r"""
    #     Info
    #     ----
    #         Aggregation of class prototypes from clients, based on tensors banks.

    #     Params
    #     ----
    #         features_banks_list (_list[torch.Tensor]_): A list of several banks of features tensors.

    #         labels_banks_list (_list[torch.Tensor]_): A list of several banks of labels tensors.
    #     """
    #     info("Aggregation For Class Prototypes From Tensors Banks", throw=False)

    #     try:
    #         assert features_banks_list and len(features_banks_list) == len(labels_banks_list), "Invalid Features Banks List Or Labels Banks List"

    #         aggregation_algorithm = wg.dict_str(self.__server._ALGO_CONFIG, "class_prototypes_aggregation_algorithm")

    #         if aggregation_algorithm == "Weighted":

    #             class_prototypes = self.__server.cal_class_prototypes(
    #                 features_bank=torch.cat(features_banks_list),
    #                 labels_bank=torch.cat(labels_banks_list)
    #             )

    #         elif aggregation_algorithm == "Average":

    #             class_prototypes_list = [self.__server.cal_class_prototypes(features_bank, labels_bank) for features_bank, labels_bank in zip(features_banks_list, labels_banks_list)]

    #             class_prototypes = torch.stack(class_prototypes_list).nanmean(dim=0)

    #         else:
    #             raise ValueError(f"Unsupported Class Prototypes Aggregation Algorithm [{aggregation_algorithm}]")

    #     except Exception as e:
    #         critical(f"{self.__server.sign} Class Prototypes Aggregation Error", e)
    #     else:
    #         self.__server.class_prototypes_sync(class_prototypes)

    # def trans_map(self, trans_maps_list: list[dict[int, Union[int, list[int]]]], labels_counts_list: Optional[list[torch.Tensor]] = None):
    #     r"""
    #     Info
    #     ----
    #         Aggregation of classes transform map for each unlearning forget class from clients. NOTE: Must have the same unlearning forget classes.

    #     Params
    #     ----
    #         trans_maps_list (_list[dict[int, int]]_): A list of several transform maps.

    #         labels_counts_list (_list[torch.Tensor]_): OPTIONAL. A list of labels_counts corresponds to trans_map, labels_counts shape as `[num_classes]`.

    #     Return
    #     ----
    #         dict[int, int]: Aggregated transform map.
    #     """
    #     info("Aggregation For Unlearning Transform Map", throw=False)

    #     try:
    #         # checking
    #         assert trans_maps_list, "No Transform Maps List"

    #         unclasses = list(trans_maps_list[0])
    #         assert unclasses == self.__server.algorithm.unclasses, f"Invalid Unlearning Forget Classes {unclasses}"

    #         trans_3D_list = [[[trans] if isinstance(trans, int) else trans for trans in trans_map.values()] for trans_map in trans_maps_list]  # shape as [groups, unclasses, keep_range]

    #         # aggregation configs
    #         aggregation_algorithm = wg.dict_str(self.__server._ALGO_CONFIG, "trans_map_aggregation_algorithm")
    #         aggregation_depth_pattern = wg.dict_str(self.__server._ALGO_CONFIG, "trans_map_aggregation_depth")

    #         aggregation_mode = None
    #         is_weighted_aggregation = False
    #         if aggregation_algorithm:
    #             aggregation_mode = aggregation_algorithm.split("_")[0]
    #             if "Weighted" in aggregation_algorithm:
    #                 is_weighted_aggregation = True

    #         info(f"Unlearning Transform Map [{aggregation_algorithm}] Aggregation For {unclasses} In [{aggregation_depth_pattern}] Depth", outline=True)

    #         max_trans_len = max([max([len(trans) for trans in trans_2D_list]) for trans_2D_list in trans_3D_list])

    #         # trans_tensors_stack
    #         trans_tensors_stack = torch.stack([wg.tensor_stack(trans_2D_list, align_len=max_trans_len) for trans_2D_list in trans_3D_list]).to(self.__server.DEVICE)  # shape as [groups, unclasses, max_trans_len]

    #         # labels_counts_stack
    #         if is_weighted_aggregation:
    #             # for weighted aggregation
    #             assert labels_counts_list, "No Labels Counts List For Weighted Aggregation"
    #             assert len(trans_maps_list) == len(labels_counts_list), "Transform Maps List And Labels Counts List Have Different Amount"
    #             labels_counts_stack = torch.stack(labels_counts_list)[:, unclasses]  # shape as [groups, unclasses]

    #         else:
    #             # for naive aggregation (all weighted of client`s class is ONE)
    #             labels_counts_stack = torch.ones(trans_tensors_stack.size(0), len(unclasses), device=self.__server.DEVICE)

    #         if aggregation_mode == "Group":
    #             # weighted select transform classes(with max count of samples) from group for each unclass.

    #             aggregated_trans_list: list[Union[int, list[int]]] = []
    #             for col, unclass in enumerate(unclasses):
    #                 # for each col (unclass)

    #                 unclass_trans_tensor = trans_tensors_stack[:, col]  # shape as [groups, max_trans_len]
    #                 unclass_labels_counts = labels_counts_stack[:, col]  # shape as [groups]

    #                 pos_unclass_trans_filter = (unclass_trans_tensor >= 0).sum(dim=1) > 0  # shape as [groups]

    #                 unique_trans_tensor, unique_trans_indices = unclass_trans_tensor[pos_unclass_trans_filter].unique(return_inverse=True, dim=0)  # shape as [unique_trans_samples, max_trans_len]; [pos_groups]
    #                 assert isinstance(unique_trans_tensor, torch.Tensor) and isinstance(unique_trans_indices, torch.Tensor)

    #                 unique_trans_counts = unique_trans_indices.bincount(unclass_labels_counts[pos_unclass_trans_filter])  # # shape as [unique_trans_samples]
    #                 assert isinstance(unique_trans_counts, torch.Tensor)

    #                 aggregated_trans = unique_trans_tensor[unique_trans_counts.argmax()]  # shape as [max_trans_len]
    #                 aggregated_trans = aggregated_trans.tolist()
    #                 if len(aggregated_trans) == 1:
    #                     # for explicit transform class
    #                     aggregated_trans = aggregated_trans[0]
    #                 else:
    #                     index = len(aggregated_trans)
    #                     while index > 0 and aggregated_trans[-1] == -1:
    #                         aggregated_trans.pop()
    #                         index -= 1

    #                 aggregated_trans_list.append(aggregated_trans)

    #                 confidence = 100 * unique_trans_counts.max() / unclass_labels_counts.sum()

    #                 info(f"Unealning Forget Class[{unclasses[col]}] Transform To [{aggregated_trans}] ({confidence}%)", end="\n" if col == len(unclasses)-1 else "", loc=False, level=2)

    #             return dict(zip(unclasses, aggregated_trans_list))

    #         elif aggregation_mode == "Class":
    #             # weighted select each transform class(with max count of samples) for each unclass.

    #             # aggregation_depth_pattern e.g.: Max_Count<=4
    #             aggregation_depth_REGEXP = re.compile(r"(?P<aggregation_depth>\w+)(<=(?P<depth_max_limit>\d*(\.\d*)?))?")
    #             try:
    #                 assert aggregation_depth_pattern is not None, "Can Not Found [trans_map_aggregation_depth]"

    #                 reg = aggregation_depth_REGEXP.search(aggregation_depth_pattern)
    #                 assert reg is not None

    #                 aggregation_depth = reg.group("aggregation_depth")

    #                 depth_max_limit_str = reg.group("depth_max_limit")
    #                 if depth_max_limit_str is not None:
    #                     if "." in depth_max_limit_str:
    #                         depth_max_limit = max(int(float(depth_max_limit_str)), 1)
    #                         warning(f"Detected Float Aggregation Depth Max Limit [{depth_max_limit_str}], Set As [{depth_max_limit}]", dim=True)
    #                     else:
    #                         depth_max_limit = int(depth_max_limit_str)
    #                         if depth_max_limit == 0:
    #                             warning("Detected Zero Aggregation Depth Max Limit, Set As [1]")
    #                             depth_max_limit = 1
    #                 else:
    #                     depth_max_limit = None

    #             except Exception as e:
    #                 raise ValueError(trace_exc("[trans_map_aggregation_depth] Parsing Error", e))

    #             aggregated_trans_list: list[Union[int, list[int]]] = []
    #             for col, unclass in enumerate(unclasses):
    #                 # for each col (unclass)

    #                 unclass_trans_tensor = trans_tensors_stack[:, col]  # shape as [groups, max_trans_len]
    #                 unclass_labels_counts = labels_counts_stack[:, col]  # shape as [groups]

    #                 pos_unclass_trans_filter = (unclass_trans_tensor >= 0).sum(dim=1) > 0  # shape as [groups]

    #                 # unclass keep range
    #                 unclass_trans_lens = torch.tensor([len(trans_2D_list[col]) for trans_2D_list in trans_3D_list], device=self.__server.DEVICE)  # shape as [groups]
    #                 unclass_depth_counts = unclass_trans_lens[pos_unclass_trans_filter].bincount(unclass_labels_counts[pos_unclass_trans_filter])  # shape as [unique_trans_lens]
    #                 assert len(unclass_depth_counts) > 0, f"No Valid Data For Unclass [{unclass}]"
    #                 if aggregation_depth == "Max":
    #                     unclass_keep_range = int(unclass_trans_lens[pos_unclass_trans_filter].max().item())

    #                 elif aggregation_depth == "Min":
    #                     unclass_keep_range = int(unclass_trans_lens[pos_unclass_trans_filter].min().item())

    #                 elif aggregation_depth == "Max_Count":
    #                     # if not is_weighted_aggregation:
    #                     #     warning("Aggregation Depth [Max_Count] For Non-Weighted Aggregation Is Same As [Max]")
    #                     unclass_keep_range = int(unclass_depth_counts.argmax().item())

    #                 else:
    #                     raise ValueError(f"Unsupported Transform Map Aggregation Depth [{aggregation_depth}]")

    #                 is_limited = False
    #                 if depth_max_limit is not None and unclass_keep_range > depth_max_limit:
    #                     unclass_keep_range = depth_max_limit
    #                     is_limited = True

    #                 keep_range_confidence = float((100 * unclass_depth_counts[unclass_keep_range] / unclass_labels_counts.sum()).item())

    #                 aggregated_depth_trans: list[int] = []
    #                 aggregated_depth_confidences: dict[int, list[float]] = {}

    #                 for depth in range(unclass_keep_range):
    #                     # for each depth

    #                     depth_unclass_trans_tensor = unclass_trans_tensor[:, depth]  # shape as [groups]

    #                     pos_depth_unclass_trans_filter = depth_unclass_trans_tensor >= 0  # shape as [groups]

    #                     if not pos_depth_unclass_trans_filter.any():
    #                         # all -1
    #                         break

    #                     unique_trans_counts = depth_unclass_trans_tensor[pos_depth_unclass_trans_filter].bincount(unclass_labels_counts[pos_depth_unclass_trans_filter])

    #                     cur_depth_trans = int(unique_trans_counts.argmax().item())
    #                     cur_depth_confidence = float((100 * unique_trans_counts.max() / unclass_labels_counts.sum()).item())

    #                     if cur_depth_trans in aggregated_depth_trans:
    #                         # if existing
    #                         aggregated_depth_confidences[cur_depth_trans].append(cur_depth_confidence)

    #                     else:
    #                         # if not existing
    #                         aggregated_depth_trans.append(cur_depth_trans)
    #                         aggregated_depth_confidences[cur_depth_trans] = [cur_depth_confidence]

    #                 aggregated_trans_list.append(aggregated_depth_trans)

    #                 confidences_text_list: list[str] = []
    #                 for confidences_list in aggregated_depth_confidences.values():
    #                     confidences_text_list.append("+".join([f"{confidence}%" for confidence in confidences_list]))

    #                 # display
    #                 unclass_depth_limit_str = "" if depth_max_limit is None else f"<={depth_max_limit}"
    #                 unclass_keep_range_str = color_str(unclass_keep_range, back="y") if is_limited else str(unclass_keep_range)
    #                 info(f"Unealning Forget Class[{unclass}] Transform To [{aggregated_depth_trans}] In {aggregation_depth} Depth [{unclass_keep_range_str}{unclass_depth_limit_str}] ({keep_range_confidence}%)", loc=False, level=2)
    #                 df = pd.DataFrame(
    #                     data={"TRANS": aggregated_depth_trans, "CONFIDENCE": confidences_text_list},
    #                     dtype=object
    #                 ).transpose().to_string(header=False)
    #                 print(df, end="\n\n" if unclass == unclasses[-1] else "\n")

    #             return dict(zip(unclasses, aggregated_trans_list))

    #         else:
    #             raise ValueError(f"Unsupported Transform Map Aggregation Algorithm [{aggregation_algorithm}]")

    #     except Exception as e:
    #         critical(f"{self.__server.sign} Nearest Class Aggregation Error", e)
    #         raise ValueError

    # # FIXME
    # def trans_map_from_tensors_banks(self, logits_banks_list: list[torch.Tensor], labels_banks_list: list[torch.Tensor]):
    #     r"""
    #     Info
    #     ----
    #         Aggregation of classes transform map for unlearning forget class from clients, based on tensors banks.

    #     Params
    #     ----
    #         logits_banks_list (_list[torch.Tensor]_): A list of several banks of logits tensors.

    #         labels_banks_list (_list[torch.Tensor]_): A list of several banks of labels tensors.

    #     Return
    #     ----
    #         _type_: _description_
    #     """
    #     info("Aggregation For Nearest Class From Tensors Banks", throw=False)

    #     try:
    #         assert logits_banks_list and len(logits_banks_list) == len(labels_banks_list), "Invalid Features Banks List Or Labels Banks List"

    #         self.__server.algorithm.eval()

    #         logits_bank = torch.cat(logits_banks_list)  # shape as [samples, num_classes]
    #         labels_bank = torch.cat(labels_banks_list)  # shape as [samples]

    #         unclass_filter = self.__server.algorithm.ul.create_formatted_tensor(labels_bank)  # shape as [samples]
    #         # nearness = int(self.__server.algorithm.args["neighbor_nearness"])  # 0: The forget class; >1: The nearness class; <0: The farness class

    #         all_unclass_logits = logits_bank[unclass_filter]  # shape as [unclasses_samples, num_classes]

    #         # top_indices
    #         _, top_indices = all_unclass_logits.topk(2, dim=1)  # shape as [unclasses_samples, num_classes]

    #         may_correct_unclass_filter = self.__server.algorithm.ul.create_formatted_tensor(top_indices[:, 0])  # may convert to other forget classes

    #         may_correct_unclass_indices = top_indices[:, 1][may_correct_unclass_filter]
    #         incorrect_unclass_indices = top_indices[:, 0][~may_correct_unclass_filter]

    #         unclass_indices = torch.cat((may_correct_unclass_indices, incorrect_unclass_indices))  # shape as [unclasses_samples]

    #         # count unlearning forget class samples predicted class
    #         unique_indices, classes_counts = unclass_indices.unique(return_counts=True)

    #         nearest_class = int(unique_indices[classes_counts.argmax()].item())
    #         counts = classes_counts.max()

    #         info(f"The Nearest Class Of Unealning Forget Class(es) {self.__server.algorithm.unclasses} Is [{nearest_class}], With Confidence Of {100.0 * counts / len(unclass_indices):.4f}%", end="\n", loc=False)

    #         return nearest_class

    #     except Exception as e:
    #         critical(f"{self.__server.sign} Nearest Class Aggregation Error", e)
