from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class AddCanvasElement(nn.Module):
    def __init__(self, discrete_fn: Optional[nn.Module] = None) -> None:
        super().__init__()

        self.x = torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.float)
        self.y = torch.tensor([0], dtype=torch.long)
        self.discrete_fn = discrete_fn

    def __call__(self, data):
        logger.debug(f"Before AddCanvasElement:\n{data}")
        if self.discrete_fn:
            data["bboxes_with_canvas"] = torch.cat(
                [self.x, self.discrete_fn.continuize(data["discrete_gold_bboxes"])],
                dim=0,
            )
        else:
            data["bboxes_with_canvas"] = torch.cat([self.x, data["bboxes"]], dim=0)
        data["labels_with_canvas"] = torch.cat([self.y, data["labels"]], dim=0)
        logger.debug(f"After AddCanvasElement:\n{data}")
        return data
