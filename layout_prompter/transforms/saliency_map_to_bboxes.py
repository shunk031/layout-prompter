from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import cv2
import numpy as np
import torch
import torch.nn as nn

if TYPE_CHECKING:
    from layout_prompter.typehint import PilImage


logger = logging.getLogger(__name__)


class SaliencyMapToBBoxes(nn.Module):
    def __init__(
        self,
        threshold: int,
        is_filter_small_bboxes: bool = True,
        min_side: int = 80,
        min_area: int = 6000,
    ) -> None:
        super().__init__()

        self.threshold = threshold
        self.is_filter_small_bboxes = is_filter_small_bboxes
        self.min_side = min_side
        self.min_area = min_area

    def _is_small_bbox(self, bbox) -> bool:
        return any(
            [
                all([bbox[2] <= self.min_side, bbox[3] <= self.min_side]),
                bbox[2] * bbox[3] < self.min_area,
            ]
        )

    def __call__(self, saliency_map: PilImage) -> torch.Tensor:
        assert saliency_map.mode == "L", "saliency map must be grayscale image"
        saliency_map_gray = np.array(saliency_map)

        _, thresholded_map = cv2.threshold(
            saliency_map_gray, self.threshold, 255, cv2.THRESH_BINARY
        )
        contours, _ = cv2.findContours(
            thresholded_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        bboxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if self.is_filter_small_bboxes and self._is_small_bbox([x, y, w, h]):
                continue
            bboxes.append([x, y, w, h])

        bboxes = sorted(bboxes, key=lambda x: (x[1], x[0]))
        bboxes = torch.tensor(bboxes)
        return bboxes
