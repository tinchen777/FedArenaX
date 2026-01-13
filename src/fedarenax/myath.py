# -*- coding: utf-8 -*-
# Python version: 3.9
# @TianZhen
r"""
Module for mathematical calculations.
"""

import torch
import torch.nn.functional as F
import numpy as np
from numpy.typing import NDArray
import logging
from torchmetrics.functional.classification import multilabel_confusion_matrix
from typing import (Union, Any, Sequence, Iterable, Optional, Literal)

# from . import wenger as wg
# from .mylog import (warning, error, trace_exc)


# def cal(*args: Union[float, str, int], **kwargs: Any):
#     r"""
#     Precise calculation of floating point numbers. Receive operator as string,

#     Parameters
#     ----------
#         *args : Union[float, str, int]
#             The operator and operands.
#             - _float_ and _int_: The operands.
#             - _str_: The operator.

#         **kwargs : Any
#             The mapping of related symbols and real values.

#     Returns
#     -------
#         float
#             The result.

#     Raises
#     ------
#         ValueError

#     Useage
#     ----
#         >>> x = (3.3-1.1)/1.1
#         x = 1.9999999999999996
#         >>> x = myath.cal("(", 3.3, "-", 1.1, ")/", 1.1)
#         x = 2.0
#     """
#     # expression
#     expression = ""
#     for symbol in args:
#         if isinstance(symbol, float):
#             symbol = f"func('{symbol}')"
#         expression += f" {symbol}"

#     try:
#         return float(wg.quick_exec(expression=f"float({expression})", use_f=False, func=Decimal, **kwargs))
#     except Exception as e:
#         raise ValueError(f"Cal [{expression}] Error", e)


# def split_int(__int: int, split: Union[int, Iterable[Union[int, float]]], random: bool = False):
#     r"""
#     Split an integer to multiple integers based on the `split`.

#     NOTE: The sum of the split integers is equal to the original integer.

#     Parameters
#     ----------
#         __int : int
#             An integer to be split.

#         split : Union[int, Iterable[Union[int, float]]]
#             The split rules.
#             - _int_: The number of parts.
#             - _Iterable_: The proportion of each part.

#         random : bool, default to `False`
#             Control whether to split randomly.
#             - `False`: Split based on the proportion after average.

#     Returns
#     -------
#         NDArray[int]
#             The split integers.
#     """
#     __int = int(__int)
#     if isinstance(split, int):
#         split = np.ones(split)
#     elif not isinstance(split, np.ndarray):
#         split = np.array(split)

#     prop_parts: NDArray[np.float64] = __int * split / split.sum()

#     integer_parts = np.floor(prop_parts).astype(np.int64)

#     remainder = __int - integer_parts.sum()

#     if remainder > 0:
#         remainder_props = prop_parts - integer_parts
#         remainder_props /= remainder_props.sum()
#         remainder_parts = np.random.choice(
#             a=len(split),
#             size=remainder,
#             replace=False,
#             p=None if random else remainder_props
#         )
#         integer_parts[remainder_parts] += 1

#     return integer_parts


# def mean(__seq: Iterable[Any]):
#     r"""
#     Calculate the mean of a sequence.

#     Parameters
#     ----------
#         __seq : Iterable[Any]
#             A sequence.

#     Returns
#     -------
#         float
#             The result.
#     """
#     arr = wg.to_array(__seq, copy=False)
#     return float(arr.mean())


# def var(__seq: Iterable[Any]):
#     r"""
#     Calculate the variance of a sequence.

#     Parameters
#     ----------
#         __seq : Iterable[Any]
#             A sequence.

#     Returns
#     -------
#         float
#             The result.
#     """
#     arr = wg.to_array(__seq, copy=False)
#     return float(arr.var())


# def std(__seq: Iterable[Any]):
#     r"""
#     Calculate the standard deviation of a sequence.

#     Parameters
#     ----------
#         __seq : Iterable[Any]
#             A sequence.

#     Returns
#     -------
#         float
#             The result.
#     """
#     arr = wg.to_array(__seq, copy=False)
#     return float(arr.std())


# def seq_exclude(__seq_a: Iterable[Any], __seq_b: Iterable[Any], assume_unique: bool = False) -> NDArray[Any]:
#     r"""
#     Get the values in the first sequence that NOT exist in the second sequence.

#     NOTE: The order is based on the first sequence.

#     Parameters
#     ----------
#         __seq_a : Iterable[Any]
#             The first sequence.

#         __seq_b : Iterable[Any]
#             The second sequence.

#         assume_unique : bool, default to `False`
#             Control whether the input sequences are assumed to be unique.
#             - `True`: The input sequences are assumed to be unique which can speed up the calculation.

#     Returns
#     -------
#         NDArray[Any]
#             The new sequence after removing.
#     """
#     arr_a = wg.to_array(__seq_a)
#     arr_b = wg.to_array(__seq_b, copy=False)
#     return arr_a[np.isin(arr_a, arr_b, invert=True, assume_unique=assume_unique)]


# def seq_include(__seq_a: Iterable[Any], __seq_b: Iterable[Any], assume_unique: bool = False) -> NDArray[Any]:
#     r"""
#     Get the values in the first sequence that exist in the second sequence.

#     NOTE: The order is based on the first sequence.

#     Parameters
#     ----------
#         __seq_a : Iterable[Any]
#             The first sequence.

#         __seq_b : Iterable[Any]
#             The second sequence.

#         assume_unique : bool, default to `False`
#             Control whether the input sequences are assumed to be unique.
#             - `True`: The input sequences are assumed to be unique which can speed up the calculation.

#     Returns
#     -------
#         NDArray[Any]
#             The new sequence after getting.
#     """
#     arr_a = wg.to_array(__seq_a)
#     arr_b = wg.to_array(__seq_b, copy=False)
#     return arr_a[np.isin(arr_a, arr_b, assume_unique=assume_unique)]


# def seq_extend(__seq_a: Iterable[Any], __seq_b: Iterable[Any], extend_tail: bool = True, assume_unique: bool = False):
#     r"""
#     Extend the first sequence with the values in the second sequence that do not exist in the first sequence.

#     NOTE: The order is based on the first sequence.

#     Parameters
#     ----------
#         __seq_a : Iterable[Any]
#             The first sequence.

#         __seq_b : Iterable[Any]
#             The second sequence.

#         extend_tail : bool, default to `True`
#             Control whether to extend at the tail of the first sequence.
#             - `False`: Extend at the head of the first sequence.

#         assume_unique : bool, default to `False`
#             Control whether the input sequences are assumed to be unique.
#             - `True`: The input sequences are assumed to be unique which can speed up the calculation.

#     Returns
#     -------
#         NDArray[Any]
#             The new sequence after extending.
#     """
#     arr_a = wg.to_array(__seq_a, copy=False)
#     arr_b = wg.to_array(__seq_b, copy=False)

#     extend_array = seq_exclude(arr_b, arr_a, assume_unique=assume_unique)
#     if extend_tail:
#         return np.concatenate((arr_a, extend_array))
#     else:
#         return np.concatenate((extend_array, arr_a))


# def deduplicate(__seq: Iterable[Any]) -> NDArray[Any]:
#     r"""
#     Remove the duplicate values in the sequence.

#     Parameters
#     ----------
#         __seq : Iterable[Any]
#             The sequence.

#     Returns
#     -------
#         NDArray[Any]
#             The new sequence after deduplicating.
#     """
#     arr = wg.to_array(__seq, copy=False)

#     _, unique_indices = np.unique(arr, return_index=True)
#     return arr[np.sort(unique_indices)]


def tensor_dict_cal(*items: Union[dict[Any, Any], int, float], operator: Literal["add", "sub", "mul", "dot", "div", "max", "min", "avg", "wt_sum", "wt_avg"], weights: Optional[Sequence[Any]] = None, strict_key: bool = False, device: Optional[torch.device] = None):
    r"""
    Calculate the operator for the same key tensor among tensor dicts.

    NOTE 1: The calculation is performed in the order of `items`.

    NOTE 2: The key of the result is based on the first dict. The data type of the result is based on the first dict.

    Parameters
    ----------
        items : Union[dict[Any, Any], int, float]
            The items (_dict_ or number) to be calculated.

        operator : Literal["add", "sub", "mul", "dot", "div", "max", "min", "avg", "wt_sum", "wt_avg"]
            The operator name.
            - `"add"`: Addition;
            - `"sub"`: Subtraction;
            - `"mul"`: Matrix Multiplication;
            - `"dot"`: Hadamard Product;
            - `"div"`: Division;
            - `"max"`: Maximum;
            - `"min"`: Minimum;
            - `"avg"`: Average, for non-empty;
            - `"wt_sum"`: Weighted sum;
            - `"wt_avg"`: Weighted average.

        weights : Optional[Sequence[Any]], default to `None`
            For `"wt_sum"` or `"wt_avg"` ONLY. The weight for each `item`.

        strict_key : bool, default to `False`
            Controls whether dict keys is strictly required.

        device : Optional[torch.device], default to `None`
            The default device for new key result. Non-tensor result will trans to a new tensor.

    Returns
    -------
        dict[Any, Tensor]
            The result dict.

    Raises
    ------
        ValueError
    """
    def calc(base: Any, target: Any, op: str):
        r"""
        Return: base [op] target
        """
        if op == "add":
            return base + target
        elif op == "sub":
            return base - target
        elif op in ["mul", "dot"]:
            if isinstance(base, torch.Tensor) and isinstance(target, torch.Tensor) and op == "mul":
                return base.matmul(target)
            else:
                return base * target
        elif op == "div":
            return base / target
        elif op == "max":
            raise NotImplementedError
        elif op == "min":
            raise NotImplementedError
        else:
            raise ValueError(f"Invalid Operator [{op}]")

    try:
        # checking
        if operator in ["wt_sum", "wt_avg"]:
            assert weights is not None, f"No Weights For Operator [{operator}]"
            assert len(items) == len(weights), f"Items [{len(items)}] And Weights [{len(weights)}] Have Different Amount"
            weights_list = weights
        else:
            weights_list = []

        # keys & start_index
        is_dict = True
        start_index = 0
        for item in items:
            if isinstance(item, dict):
                if item:
                    keys = list(item)
                    break
                else:
                    # empty dict
                    if is_dict:
                        start_index += 1
            else:
                is_dict = False
        else:
            raise TypeError("Can Not Found Any Dict")

        result: dict[Any, torch.Tensor] = {}
        dtype = None

        for key in keys:
            # for each key

            # first item
            start_item = items[start_index]  # not a empty dict
            if isinstance(start_item, dict):
                base_val = start_item[key]  # first dict
            else:
                base_val = start_item

            # base count
            if operator in ["wt_sum", "wt_avg"]:
                base_weight = weights_list[start_index]
                base_val = calc(base_val, base_weight, op="dot")
                count = base_weight
            else:
                count = 1

            for index, item in enumerate(items[start_index+1:], start=start_index+1):
                # for remain item

                # taget value
                if isinstance(item, dict):
                    if item:
                        try:
                            target_val = item[key]
                        except Exception:
                            if strict_key:
                                raise KeyError(f"Can Not Found Key [{key}] In Dict At Position [{index}]")
                            else:
                                continue
                    else:
                        # empty dict
                        continue
                else:
                    target_val = item

                # update count
                if operator in ["wt_sum", "wt_avg"]:
                    target_weight = weights_list[index]
                    target_val = calc(target_val, target_weight, op="dot")
                    count = count + target_weight
                else:
                    count = count + 1

                # dtype
                if isinstance(base_val, torch.Tensor):
                    dtype = base_val.dtype
                elif isinstance(target_val, torch.Tensor):
                    dtype = target_val.dtype
                else:
                    dtype = None

                # calculation
                try:
                    base_val = calc(base_val, target_val, op="add" if operator in ["avg", "wt_sum", "wt_avg"] else operator)
                except Exception as e:
                    raise ValueError(f"[{operator}] Calculation Error For Key [{key}] Of Item At Position [{index}]", e)

                if dtype is not None and isinstance(base_val, torch.Tensor):
                    base_val = base_val.to(dtype=dtype)

            # for avg
            if operator in ["avg", "wt_avg"]:
                base_val = base_val / count

            if dtype is not None and isinstance(base_val, torch.Tensor):
                base_val = base_val.to(dtype=dtype)

            if not isinstance(base_val, torch.Tensor):
                base_val = torch.tensor(base_val, device=device)

            result[key] = base_val

        return result

    except Exception as e:
        raise ValueError("Tensor Dict Calculation Error", e)


def CM_to_CCM(CM: torch.Tensor):
    r"""
    Convert `confusion matrix (CM)` to `class confusion matrix (CCM)`.

    Parameters
    ----------
        CM : torch.Tensor
            A `confusion matrix`.

    Returns
    -------
        torch.Tensor
            `class confusion matrix`.

    Raises
    ------
        ValueError
    """
    try:
        # checking
        assert isinstance(CM, torch.Tensor), "Invalid Confusion Matrix"
        try:
            assert CM.size(0) == CM.size(1) and CM.dim() == 2
        except Exception:
            raise ValueError(f"Invalid Shape Of Confusion Matrix [{CM.size()}]")

        sum_rows = CM.sum(dim=1)
        sum_columns = CM.sum(dim=0)
        CM_sum = CM.sum()

        class_CMs: list[torch.Tensor] = []

        for i in range(CM.size(0)):

            TP = CM[i, i]

            FN = sum_rows[i] - TP

            FP = sum_columns[i] - TP

            TN = CM_sum - TP - FN - FP

            class_CMs.append(torch.tensor([[TN, FP], [FN, TP]]))

        return torch.stack(class_CMs).to(CM.device)

    except Exception as e:
        raise ValueError("Convert Confusion Matrix To Class Confusion Matrix Error", e)


class MetricFromMatrix(object):
    r"""
    Class of metrics calculation based on `class confusion matrix`.
    """
    @classmethod
    def from_data(cls, preds: torch.Tensor, targets: torch.Tensor, num_classes: int):
        r"""
        Initialize `MetricFromMatrix` from `preds` and `targets`.

        Parameters
        ----------
            preds : torch.Tensor
                A tensor of predicted values. Shape as `[num_samples]`.

            targets : torch.Tensor
                A tensor of target values. Shape as `[num_samples]`.

            num_classes : int
                The number of classes.

        Raises
        ------
            ValueError
        """
        try:
            # preds & targets
            assert isinstance(preds, torch.Tensor) and isinstance(targets, torch.Tensor) and preds.size(0) == targets.size(0), f"Invalid Predicteds [shape: {preds.size()}] Or Targets [shape: {targets.size()}]"

            # num_classes
            assert isinstance(num_classes, int) and num_classes >= 2, f"Invalid Number Of Classes [{num_classes}]"

            MLCM = multilabel_confusion_matrix(
                preds=F.one_hot(preds, num_classes=num_classes),
                target=F.one_hot(targets, num_classes=num_classes),
                num_labels=num_classes
            )
            return cls(class_confusion_matrix=MLCM)

        except Exception as e:
            raise ValueError("[MetricFromMatrix] Initialization From Data Error", e)

    @classmethod
    def from_confusion_matrix(cls, confusion_matrix: torch.Tensor):
        r"""
        Initialize `MetricFromMatrix` from `confusion matrix`.

        Parameters
        ----------
            confusion_matrix : torch.Tensor
                A tensor of confusion matrix. Shape as `[num_classes, num_classes]`.

        Raises
        ------
            ValueError
        """
        try:
            return cls(class_confusion_matrix=CM_to_CCM(confusion_matrix))
        except Exception as e:
            raise ValueError("[MetricFromMatrix] Initialization From Confusion Matrix Error", e)

    def __init__(self, class_confusion_matrix: torch.Tensor):
        r"""
        Initialize `MetricFromMatrix`.

        NOTE: Calculated intermediate parameters are automatically saved to reduce subsequent calculation overhead.

        Parameters
        ----------
            class_confusion_matrix : torch.Tensor
                A tensor of `confusion_matrix` split by class. Shape as `[num_classes, 2, 2]`.

                For each `confusion_matrix`:
                - `CM[0, 0]`: true_negatives(TN);
                - `CM[0, 1]`: false_positives(FP);
                - `CM[1, 0]`: false_negatives(FN);
                - `CM[1, 1]`: true_positives(TP).

        Raises
        ------
            ValueError
        """
        self.__conventional_f1_score = False

        self.__metric: Optional[str] = None
        self.__average: Optional[str] = None
        try:
            # checking class_confusion_matrix
            assert isinstance(class_confusion_matrix, torch.Tensor), "Invalid Class Confusion Matrix"
            assert class_confusion_matrix.size(1) == 2 and class_confusion_matrix.size(2) == 2, f"Invalid Class Confusion Matrix [Shape:{class_confusion_matrix.size()}]"
            self.__class_confusion_matrix = class_confusion_matrix

            # confusion_matrix_sum. for Micro Average. shape: [2, 2]
            self.__confusion_matrix_sum = class_confusion_matrix.sum(dim=0)

            # class_weights. for Weighted Average. shape: [num_classes]
            self.__class_weights = class_confusion_matrix[:, 1].sum(dim=1) / class_confusion_matrix[0].sum()

        except Exception as e:
            raise ValueError("[MetricFromMatrix] Initialization Error", e)

        # middle parameters
        self.__class_recalls = None  # shape: [num_classes]
        self.__class_precisions = None  # shape: [num_classes]
        self.__class_f1_scores = None  # shape: [num_classes]
        self.__micro_recall = None  # shape: SCALAR
        self.__micro_precision = None  # shape: SCALAR

    def calculate(self, metric: Literal["precision", "recall", "f1_score"] = "f1_score", average: Literal["macro", "micro", "weighted", None] = "micro", conventional_f1_score: bool = False):
        r"""
        Calculate the given metric for each class and average based on the given average name.

        NOTE 1: Calculated intermediate parameters are automatically saved to reduce subsequent calculation overhead.

        NOTE 2: Calculation errors in calculations on different devices (CPU and GPU) for `macro` and `weighted`, about `e-8`.

        Parameters
        ----------
            metric : Literal["precision", "recall", "f1_score"], default to `"f1_score"`
                A name of metric. Including:
                - `"precision"`: TP / (TP + FP);
                - `"recall"`: TP / (TP + FN);
                - `"f1_score"`: 2 * precision * recall / (precision + recall).

            average : Literal["macro", "micro", "weighted", None], default to `"micro"`
                A name of average. Including:
                - `"macro"`: e.g.: mean(precision_A, precision_B, ...);
                - `"micro"`: e.g.: sum(TP_A, TP_B, ...) / (sum(TP_A, TP_B, ...) + sum(FP_A, FP_B, ...));
                - `"weighted"`: The class weights of `class_confusion_matrix` e.g.: precision_A * W_A + precision_B * W_B + ... i.e. for each class `W_a = (TRUE labels for a) / (total samples)`;
                - `None`: return the metric for each class without average.

            conventional_f1_score : bool, default to `False`
                Control whether to calculate `f1_score` in conventional way. Specifically, if both `precision` and `recall` are 0.0, the `f1_score` is 0.0 instead of `nan`.

                NOTE: If `precision` or `recall` is `nan`, the `f1_score` is still `nan`.

        Returns
        -------
            torch.Tensor
                shape: SCALAR if `average` is `"macro"`, `"weighted"` or `"micro"`;
                shape: [num_classes] if`average` is `None`.

        Raises
        ------
            ValueError
        """
        self.__conventional_f1_score = conventional_f1_score
        metric_str = "?"

        try:
            # checking metric
            assert metric in ["precision", "recall", "f1_score"], f"Invalid Metric Name [{metric}]"

            metric_str = metric.title()

            if average == "macro" or average == "weighted" or average is None:
                # I. class_metrics
                if metric == "precision":
                    # class_precisions
                    class_metrics = self.__cal_class_precisions()
                elif metric == "recall":
                    # class_recalls
                    class_metrics = self.__cal_class_recalls()
                else:
                    # class_f1_scores
                    class_metrics = self.__cal_class_f1_scores()

                # II. average
                if average == "macro":
                    # macro average
                    result = class_metrics.nanmean()
                elif average == "weighted":
                    # weighted average
                    result = (class_metrics * self.__class_weights).nansum(dim=0)
                else:
                    # no average
                    result = class_metrics

            elif average == "micro":
                if metric == "precision":
                    # micro_precision
                    result = self.__cal_micro_precision()
                elif metric == "recall":
                    # micro_recall
                    result = self.__cal_micro_recall()
                else:
                    # micro_f1_score
                    result = self.__cal_micro_f1_score()

            else:
                raise ValueError(f"Invalid Average Name [{average}]")

            self.__metric = metric
            self.__average = average

            return result

        except Exception as e:
            avg_str = f"{average.title()}-" if average and isinstance(average, str) else ""
            raise ValueError(f"[metric_from_matrix()] Error For [{avg_str}{metric_str}]", e)

    def calculate_from_string(self, __metric_str: str, split_tag: str = "_", conventional_f1_score: bool = False):
        r"""
        Calculate the given metric for each class and average based on the given metric description string.

        Parameters
        ----------
            __metric_str : str
                A metric description string.
                Support `"class_precisions"`, `"micro_precision"`, `"macro_precision"`, `"weighted_precision"`, `"class_recalls"`, `"micro_recall"`, `"macro_recall"`, `"weighted_recall"`, `"class_f1_scores"`, `"micro_f1_score"`, `"macro_f1_score"`, `"weighted_f1_score"`.

            split_tag : str, default to `"_"`
                The split tag for `average` and `metric`, split only once from the left.

            conventional_f1_score : bool, default to `False`
                Control whether to calculate `f1_score` in conventional way. Specifically, if both `precision` and `recall` are 0.0, the `f1_score` is 0.0 instead of `nan`.

                NOTE: If `precision` or `recall` is `nan`, the `f1_score` is still `nan`.

        Returns
        -------
            torch.Tensor
                shape: SCALAR if `average` is `"macro"`, `"weighted"` or `"micro"`;
                shape: [num_classes] if`average` is `None`.

        Raises
        ------
            ValueError
        """
        try:
            # checking metric_str
            assert __metric_str and isinstance(__metric_str, str), f"Invalid Metric String For Calculating [{__metric_str}]"
            # checking split_tag
            assert split_tag and isinstance(split_tag, str), f"Invalid Split Tag For Metric String Splitting [{split_tag}]"

            metric_str_split = __metric_str.split(split_tag, maxsplit=1)

            # average_name
            average_name = metric_str_split[0]
            if average_name == "class":
                average_name = None
            assert average_name == "macro" or average_name == "micro" or average_name == "weighted" or average_name is None, f"Invalid Average Name [{average_name}]"

            # metric_name
            metric_name = metric_str_split[1]
            metric_name = metric_name.strip("s")
            assert metric_name == "precision" or metric_name == "recall" or metric_name == "f1_score", f"Invalid Metric Name [{metric_name}]"

            return self.calculate(metric=metric_name, average=average_name, conventional_f1_score=conventional_f1_score)

        except Exception as e:
            raise ValueError("[calculate_from_string()] Error", e)

    @property
    def class_weights(self):
        r"""
        The class weights of `class_confusion_matrix`, shape as `[num_classes]`. i.e. for each class `W_a = (TRUE labels for a) / (total samples)`
        """
        return self.__class_weights

    @property
    def metric_name(self):
        r"""
        The formatted metric name for last `calculate()`.
        """
        if self.__metric:
            # average
            if self.__average:
                result = f"{self.__average}_{self.__metric}"
            else:
                result = f"class_{self.__metric}s"
        else:
            result = ""

        return result

    def __cal_class_precisions(self):
        r"""
        Calculate `class precisions` and store the result, try using the existing result.

        Returns
        -------
            torch.Tensor
                Class precisions, shape as `[num_classes]`.
        """
        if self.__class_precisions is None:
            TP = self.__class_confusion_matrix[:, 1, 1]
            FP = self.__class_confusion_matrix[:, 0, 1]
            self.__class_precisions = TP / (TP + FP)

        return self.__class_precisions

    def __cal_class_recalls(self):
        r"""
        Calculate `class recalls` and store the result, try using the existing result.

        Returns
        -------
            torch.Tensor
                Class recalls, shape as `[num_classes]`.
        """
        if self.__class_recalls is None:
            TP = self.__class_confusion_matrix[:, 1, 1]
            FN = self.__class_confusion_matrix[:, 1, 0]
            self.__class_recalls = TP / (TP + FN)

        return self.__class_recalls

    def __cal_class_f1_scores(self):
        r"""
        Calculate `class F1 scores` based on `class_precisions` or `class_recalls` and store the result, try using the existing result.

        Returns
        -------
            torch.Tensor
                Class F1 scores, shape as `[num_classes]`.
        """
        if self.__class_f1_scores is None:
            P = self.__cal_class_precisions()
            R = self.__cal_class_recalls()
            self.__class_f1_scores = 2 * P * R / (P + R)

            if self.__conventional_f1_score:
                self.__class_f1_scores = torch.tensor(0.0, device=P.device).where(
                    condition=(P == 0.0) & (R == 0.0),
                    other=self.__class_f1_scores
                )

        return self.__class_f1_scores

    def __cal_micro_precision(self):
        r"""
        Calculate `micro precision` and store the result, try using the existing result.

        Returns
        -------
            torch.Tensor
                Micro precision, shape as `SCALAR`.
        """
        if self.__micro_precision is None:
            TP = self.__confusion_matrix_sum[1, 1]
            FP = self.__confusion_matrix_sum[0, 1]
            self.__micro_precision = TP / (TP + FP)

        return self.__micro_precision

    def __cal_micro_recall(self):
        r"""
        Calculate `micro recall` and store the result, try using the existing result.

        Returns
        -------
            torch.Tensor
                Micro recall, shape as `SCALAR`.
        """
        if self.__micro_recall is None:
            TP = self.__confusion_matrix_sum[1, 1]
            FN = self.__confusion_matrix_sum[1, 0]
            self.__micro_recall = TP / (TP + FN)

        return self.__micro_recall

    def __cal_micro_f1_score(self):
        r"""
        Calculate `micro F1 scores` based on `micro_precision` or `micro_recall`.

        Returns
        -------
            torch.Tensor
                Micro F1 scores, shape as `SCALAR`.
        """
        P = self.__cal_micro_precision()
        R = self.__cal_micro_recall()

        if self.__conventional_f1_score and (P == 0.0 and R == 0.0):
            micro_f1_score = torch.tensor(0.0, device=P.device)
        else:
            micro_f1_score = 2 * P * R / (P + R)

        return micro_f1_score

    def __call__(self, __metric_str: str, split_tag: str = "_", conventional_f1_score: bool = False):
        r"""
        Call `calculate_from_string()`
        """
        return self.calculate_from_string(__metric_str, split_tag=split_tag, conventional_f1_score=conventional_f1_score)


def metrics_from_matrix(__matrix: torch.Tensor, cal_content: list[str], sort: bool = False, metric_name_func: Optional[Any] = None, conventional_f1_score: bool = False):
    r"""
    Calculate metrics based on `class confusion matrix` or `confusion matrix`.

    Parameters
    ----------
        __matrix : torch.Tensor
            A matrix is a `class confusion matrix` or a `confusion matrix`, according to the dimension of the matrix.
            As a `class confusion matrix`, shape as `[num_classes, 2, 2]`. For each `confusion_matrix`:
            - `CM[0, 0]`: true_negatives(TN);
            - `CM[0, 1]`: false_positives(FP);
            - `CM[1, 0]`: false_negatives(FN);
            - `CM[1, 1]`: true_positives(TP).

            As `confusion matrix`, shape as a `[num_classes, num_classes]`.

        cal_content : list[str]
            The content of metric strings.
            Support `"class_precisions"`, `"micro_precision"`, `"macro_precision"`, `"weighted_precision"`, `"class_recalls"`, `"micro_recall"`, `"macro_recall"`, `"weighted_recall"`, `"class_f1_scores"`, `"micro_f1_score"`, `"macro_f1_score"`, `"weighted_f1_score"`.

        sort : bool, default to `False`
            Control whether sort the `cal_content`.

        metric_name_func : Optional[Any], default to `None`
            The function for handling metric name in result dict. e.g.: lambda name: f"cur_{name}"

        conventional_f1_score : bool, default to `False`
            Control whether to calculate `f1_score` in conventional way. Specifically, if both `precision` and `recall` are 0.0, the `f1_score` is 0.0 instead of `nan`.

            NOTE: If `precision` or `recall` is `nan`, the `f1_score` is still `nan`.

    Returns
    -------
        dict[str, torch.Tensor]
            A _dict_ of valid metric name to value.
    """
    metrics_dict: dict[str, torch.Tensor] = {}
    try:
        # judge matrix
        if __matrix.dim() == 2:
            # try as confusion matrix
            metric_cal = MetricFromMatrix.from_confusion_matrix(__matrix)
        else:
            # try as class confusion matrix
            metric_cal = MetricFromMatrix(__matrix)

        # cal_content
        assert isinstance(cal_content, list) and cal_content, f"Invalid Calculate Content [{cal_content}]"
        if sort:
            cal_content = list(set(cal_content))
            # FIXME
            cal_content.sort()  # class_precisions, class_recalls在最前

        for cal_item in cal_content:
            # for each cal
            try:
                metric_val = metric_cal(cal_item, conventional_f1_score=conventional_f1_score)

                # metric_name
                metric_name = metric_cal.metric_name
                if metric_name_func is not None:
                    metric_name = metric_name_func(metric_name)
                # save
                metrics_dict[metric_name] = metric_val

            except Exception as e:
                logging.warning(f"Metric [{cal_item}] Calculate Error", e)

        assert metrics_dict, f"No Metric Create From Calculate Content [{cal_content}]"

    except Exception as e:
        logging.error("[metrics_from_matrix()] Error", e)

    return metrics_dict


# class TF_IDF(object):
#     r"""
#     Class of `TF-IDF` algorithm.

#     NOTE: `TF`: Term Frequency. `IDF`: Inverse Document Frequency.
#     """
#     def __init__(self, mode: Literal["value", "count"] = "count", term_as_position: bool = False):
#         r"""
#         Initialize `TF_IDF`.

#         Parameters
#         ----------
#             mode : Literal["value", "count"], default to `"count"`
#                 The calculation mode.
#                 - `"value"`: Calculation based on value of term;
#                 - `"count"`: Calculation based on count of term.

#             term_as_position : bool, default to `False`
#                 Control whether to consider term as a position in document instead of value in document.
#         """
#         self.__mode = mode
#         self.__term_as_position = bool(term_as_position)

#     def cal(self, ts: Iterable[Any], d: Iterable[Any], D: Iterable[Any], out_type_as: Optional[Literal["ts", "d", "D"]] = None):
#         r"""
#         Calculate `TF-IDF` score for each term `t` in `ts`. `t` is in document `d` from the document set `D`.

#         Parameters
#         ----------
#             ts : Iterable[Any]
#                 A set of terms.

#             d : Iterable[Any]
#                 A specific document.

#             D : Iterable[Any]
#                 A set of document, shape as `[num_d, ...]`.

#             out_type_as : Optional[Literal["ts", "d", "D"]], default to `None`
#                 The output type.
#                 - `None`: `NDArray[Any]`;
#                 - Others: As `ts`, `d`, `D`.

#         Returns
#         -------
#             NDArray[np.float]
#                 TF-IDF.

#         Raises
#         ------
#             ValueError
#         """
#         try:
#             TFs = self.cal_TF(ts=ts, d=d)  # shape as [ts]
#             IDFs = self.cal_IDF(ts=ts, D=D)  # shape as [ts]
#             result = TFs * IDFs  # shape as [ts]

#             if out_type_as:
#                 dtype = type(eval(out_type_as))
#                 if issubclass(dtype, list):
#                     result = result.tolist()
#                 elif issubclass(dtype, torch.Tensor):
#                     result = torch.tensor(result, device=eval(f"{out_type_as}.device"))

#             return result

#         except Exception as e:
#             raise ValueError("Calculate TF-IDF Error", e)

#     def cal_TF(self, ts: Iterable[Any], d: Iterable[Any]):
#         r"""
#         Calculate `Term Frequency`, represents the contribution of each term `t` to a specific document `d`.

#         Parameters
#         ----------
#             ts : Iterable[Any]
#                 A set of terms.

#             d : Iterable[Any]
#                 A specific document.

#         Returns
#         -------
#             NDArray[np.float]
#                 TF.

#         Raises
#         ------
#             ValueError
#         """
#         try:
#             # terms
#             ts_arr = wg.to_array(ts)
#             assert len(ts_arr.shape) == 1, f"Terms [ts] Must Be 1-D Not [{len(ts_arr.shape)}-D]"
#             # document
#             d_arr = wg.to_array(d)
#             assert len(d_arr.shape) == 1, f"Document [d] Must Be 1-D Not [{len(d_arr.shape)}-D]"

#             if self.__term_as_position:
#                 # for term as position
#                 try:
#                     ts_arr = d_arr[ts_arr]
#                     assert isinstance(ts_arr, np.ndarray)
#                 except Exception as e:
#                     raise IndexError("Calculate Term As Position Error", e))

#             if self.__mode == "value":
#                 r"""
#                 calculate TF in value mode
#                 TF = term_value / document_value_sum
#                 """
#                 if np.issubdtype(ts_arr.dtype, np.str_) or np.issubdtype(d_arr.dtype, np.str_):
#                     raise TypeError(f"Can Not Calculate [{ts_arr.dtype}] Terms Or [{d_arr.dtype}] Document Based On Value")

#                 TF: NDArray[np.float_] = ts_arr / d_arr.sum()  # shape as [ts_arr]

#             elif self.__mode == "count":
#                 r"""
#                 calculate TF in count mode
#                 TF = term_count_in_document / document_words_amount
#                 """
#                 TF_list: list[float] = []
#                 d_sum = len(d_arr)

#                 for t in ts_arr:
#                     t_count = np.count_nonzero(t == d_arr)

#                     TF_list.append(t_count / d_sum)

#                 TF = np.array(TF_list)  # shape as [ts_arr]

#             else:
#                 raise ValueError(f"Invalid Calculate Mode [{self.__mode}]")

#             return TF

#         except Exception as e:
#             raise ValueError("Calculate TF(Term Frequency) Error", e))

#     def cal_IDF(self, ts: Iterable[Any], D: Iterable[Any]):
#         r"""
#         Calculate `Inverse Document Frequency`, represents how common or rare contribution of a specific term `t` is in the entire document set `D`.

#         NOTE: The closer it is to 0, the more common contribution of a term is.

#         Parameters
#         ----------
#             ts : Iterable[Any]
#                 A set of terms.

#             D : Iterable[Any]
#                 A set of document, shape as `[num_d, ...]`.

#         Returns
#         -------
#             NDArray[np.float]
#                 IDF.

#         Raises
#         ------
#             ValueError
#         """
#         try:
#             # terms
#             ts_arr = wg.to_array(ts)
#             assert len(ts_arr.shape) == 1, f"Terms [ts] Must Be 1-D Not [{len(ts_arr.shape)}-D]"
#             # documents
#             if not isinstance(D, list):
#                 D = wg.to_array(D)
#                 assert len(D.shape) == 2, f"Documents [D] Must Be 2-D Not [{len(D.shape)}-D]"
#             D = [wg.to_array(d, copy=False) for d in D]
#             is_str_D = True in [np.issubdtype(d.dtype, np.str_) for d in D]

#             # sum of D
#             D_sum = len(D)

#             # count of d
#             if self.__mode == "value":
#                 if np.issubdtype(ts_arr.dtype, np.str_) or is_str_D:
#                     raise TypeError(f"Can Not Calculate [{ts_arr.dtype}] Terms Or [{'String' if is_str_D else 'Non-String'}] Documents Based On Value")

#                 if self.__term_as_position:
#                     # for term as position
#                     try:
#                         t_vals = np.stack([d[ts_arr] for d in D], axis=1)  # shape as [ts_arr, D]
#                     except Exception as e:
#                         raise IndexError("Calculate Term As Position Error", e))
#                 else:
#                     t_vals = np.tile(np.expand_dims(ts_arr, 1), D_sum)  # shape as [ts_arr, D]

#                 d_means = np.array([d.mean() for d in D])  # shape as [D]

#                 d_counts: NDArray[Any] = np.count_nonzero(t_vals >= d_means, axis=1)  # shape as [ts_arr]

#             elif self.__mode == "count":
#                 if self.__term_as_position:
#                     raise ValueError("Can Not Calculate Term As Position Baesd On Count")

#                 d_counts = np.zeros_like(ts_arr)  # shape as [ts_arr]

#                 for d in D:
#                     d_counts += np.isin(ts_arr, d)

#             else:
#                 raise ValueError(f"Invalid Calculate Mode [{self.__mode}]")

#             r"""
#             calculate IDF
#             IDF = log[(1 + num_document) / (1 + num_document_include_term)]
#             """

#             return np.log((1 + D_sum) / (1 + d_counts))  # shape as [ts_arr]

#         except Exception as e:
#             raise ValueError("Calculate IDF(Inverse Document Frequency) Error", e))

#     def __call__(self, ts: Iterable[Any], d: Iterable[Any], D: Iterable[Any], out_type_as: Optional[Literal["ts", "d", "D"]] = None):
#         r"""
#         Implement `cal()`.
#         """
#         return self.cal(ts=ts, d=d, D=D, out_type_as=out_type_as)
