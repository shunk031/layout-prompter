from __future__ import annotations

import copy
import logging
import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from layout_prompter.utils import decapulate

if TYPE_CHECKING:
    from layout_prompter.typehint import ProcessedLayoutData

logger = logging.getLogger(__name__)


class DiscretizeBoundingBox(nn.Module):
    def __init__(self, num_x_grid: int, num_y_grid: int) -> None:
        super().__init__()

        self.num_x_grid = num_x_grid
        self.num_y_grid = num_y_grid
        self.max_x = self.num_x_grid
        self.max_y = self.num_y_grid

    def discretize(self, bbox: torch.Tensor) -> torch.Tensor:
        """
        Args:
            continuous_bbox torch.Tensor: N * 4
        Returns:
            discrete_bbox torch.LongTensor: N * 4
        """
        cliped_boxes = torch.clip(bbox, min=0.0, max=1.0)
        x1, y1, x2, y2 = decapulate(cliped_boxes)
        discrete_x1 = torch.floor(x1 * self.max_x)
        discrete_y1 = torch.floor(y1 * self.max_y)
        discrete_x2 = torch.floor(x2 * self.max_x)
        discrete_y2 = torch.floor(y2 * self.max_y)
        return torch.stack(
            [discrete_x1, discrete_y1, discrete_x2, discrete_y2], dim=-1
        ).long()

    def continuize(self, bbox: torch.Tensor) -> torch.Tensor:
        """
        Args:
            discrete_bbox torch.LongTensor: N * 4

        Returns:
            continuous_bbox torch.Tensor: N * 4
        """
        x1, y1, x2, y2 = decapulate(bbox)
        cx1, cx2 = x1 / self.max_x, x2 / self.max_x
        cy1, cy2 = y1 / self.max_y, y2 / self.max_y
        return torch.stack([cx1, cy1, cx2, cy2], dim=-1).float()

    def continuize_num(self, num: int) -> float:
        return num / self.max_x

    def discretize_num(self, num: float) -> int:
        return int(math.floor(num * self.max_y))

    def __call__(self, data: ProcessedLayoutData) -> ProcessedLayoutData:
        logger.debug(f"Before DiscretizeBoundingBox:\n{data}")
        if "gold_bboxes" not in data.keys():
            data["gold_bboxes"] = copy.deepcopy(data["bboxes"])

        data["discrete_bboxes"] = self.discretize(data["bboxes"])
        data["discrete_gold_bboxes"] = self.discretize(data["gold_bboxes"])
        if "content_bboxes" in data.keys():
            data["discrete_content_bboxes"] = self.discretize(data["content_bboxes"])

        logger.debug(f"After DiscretizeBoundingBox:\n{data}")
        return data
