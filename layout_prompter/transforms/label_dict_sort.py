from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING, Dict, Optional

import torch.nn as nn

if TYPE_CHECKING:
    from layout_prompter.typehint import ProcessedLayoutData

logger = logging.getLogger(__name__)


class LabelDictSort(nn.Module):
    """
    sort elements in one layout by their label
    """

    def __init__(self, index2label: Optional[Dict[int, str]]) -> None:
        super().__init__()
        assert index2label is not None
        self.index2label = index2label

    def __call__(self, data: ProcessedLayoutData) -> ProcessedLayoutData:
        logger.debug(f"Before LabelDictSort:\n{data}")

        # NOTE: for refinement
        if "gold_bboxes" not in data.keys():
            data["gold_bboxes"] = copy.deepcopy(data["bboxes"])

        labels = data["labels"].tolist()
        idx2label = [[idx, self.index2label[labels[idx]]] for idx in range(len(labels))]
        idx2label_sorted = sorted(idx2label, key=lambda x: x[1])  # type: ignore
        idx_sorted = [d[0] for d in idx2label_sorted]
        data["bboxes"], data["labels"] = (
            data["bboxes"][idx_sorted],
            data["labels"][idx_sorted],
        )
        data["gold_bboxes"] = data["gold_bboxes"][idx_sorted]

        logger.debug(f"After LabelDictSort:\n{data}")

        return data
