from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING

import torch.nn as nn

if TYPE_CHECKING:
    from layout_prompter.typehint import ProcessedLayoutData

logger = logging.getLogger(__name__)


class LexicographicSort(nn.Module):
    """
    sort elements in one layout by their top and left postion
    """

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, data: ProcessedLayoutData) -> ProcessedLayoutData:
        if "gold_bboxes" not in data.keys():
            data["gold_bboxes"] = copy.deepcopy(data["bboxes"])
        try:
            left, top, _, _ = data["bboxes"].t()
        except Exception as err:
            logger.debug(data["bboxes"])
            raise err
        _zip = zip(*sorted(enumerate(zip(top, left)), key=lambda c: c[1:]))
        idx = list(list(_zip)[0])
        data["ori_bboxes"], data["ori_labels"] = data["gold_bboxes"], data["labels"]
        data["bboxes"], data["labels"] = data["bboxes"][idx], data["labels"][idx]
        data["gold_bboxes"] = data["gold_bboxes"][idx]
        return data
