from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, TypedDict

import torch

from layout_prompter.utils import (
    compute_alignment,
    compute_maximum_iou,
    compute_overlap,
    convert_ltwh_to_ltrb,
)

if TYPE_CHECKING:
    from layout_prompter.parsers import ParserOutput

__all__ = ["Ranker", "RankerOutput"]


class RankerOutput(TypedDict):
    bboxes: torch.Tensor
    labels: torch.Tensor


@dataclass
class Ranker(object):
    lambda_1: float = 0.2
    lambda_2: float = 0.2
    lambda_3: float = 0.6

    val_dataset: Optional[List[Dict[str, Any]]] = field(repr=False, default=None)
    _val_bboxes: Optional[List[torch.Tensor]] = field(repr=False, default=None)
    _val_labels: Optional[List[torch.Tensor]] = field(repr=False, default=None)

    def __post_init__(self) -> None:
        assert self.lambda_1 + self.lambda_2 + self.lambda_3 == 1.0

        if self.val_dataset is None:
            return

        self._val_bboxes = [vd["bboxes"] for vd in self.val_dataset]
        self._val_labels = [vd["labels"] for vd in self.val_dataset]

    @property
    def val_bboxes(self) -> List[torch.Tensor]:
        assert self._val_bboxes is not None
        return self._val_bboxes

    @property
    def val_labels(self) -> List[torch.Tensor]:
        assert self._val_labels is not None
        return self._val_labels

    def __call__(self, predictions: List[ParserOutput]) -> List[RankerOutput]:
        metrics = []

        for prediction in predictions:
            pred_bboxes = prediction["bboxes"]
            pred_labels = prediction["labels"]

            metric = []
            _pred_labels = pred_labels.unsqueeze(0)
            _pred_bboxes = convert_ltwh_to_ltrb(pred_bboxes).unsqueeze(0)
            _pred_padding_mask = torch.ones_like(_pred_labels).bool()
            metric.append(compute_alignment(_pred_bboxes, _pred_padding_mask))
            metric.append(compute_overlap(_pred_bboxes, _pred_padding_mask))

            if self.val_dataset:
                metric.append(
                    compute_maximum_iou(
                        pred_labels, pred_bboxes, self.val_labels, self.val_bboxes
                    )
                )
            metrics.append(metric)

        metrics_tensor = torch.tensor(metrics)
        min_vals, _ = torch.min(metrics_tensor, 0, keepdim=True)
        max_vals, _ = torch.max(metrics_tensor, 0, keepdim=True)
        scaled_metrics = (metrics_tensor - min_vals) / (max_vals - min_vals)
        if self.val_dataset:
            quality = (
                scaled_metrics[:, 0] * self.lambda_1
                + scaled_metrics[:, 1] * self.lambda_2
                + (1 - scaled_metrics[:, 2]) * self.lambda_3
            )
        else:
            quality = (
                scaled_metrics[:, 0] * self.lambda_1
                + scaled_metrics[:, 1] * self.lambda_2
            )
        _predictions = sorted(zip(predictions, quality), key=lambda x: x[1])
        ranked_predictions = [item[0] for item in _predictions]
        return ranked_predictions
