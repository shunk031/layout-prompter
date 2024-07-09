from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING

import numpy as np
import torch.nn as nn

if TYPE_CHECKING:
    from layout_prompter.typehint import ProcessedLayoutData

logger = logging.getLogger(__name__)


class ShuffleElements(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, data: ProcessedLayoutData) -> ProcessedLayoutData:
        logger.debug(f"Before ShuffleElements:\n{data}")
        if "gold_bboxes" not in data.keys():
            data["gold_bboxes"] = copy.deepcopy(data["bboxes"])

        ele_num = len(data["labels"])
        shuffle_idx = np.arange(ele_num)
        np.random.shuffle(shuffle_idx)
        data["bboxes"] = data["bboxes"][shuffle_idx]
        data["gold_bboxes"] = data["gold_bboxes"][shuffle_idx]
        data["labels"] = data["labels"][shuffle_idx]
        logger.debug(f"After ShuffleElements:\n{data}")
        return data
