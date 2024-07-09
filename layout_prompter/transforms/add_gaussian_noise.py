from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from layout_prompter.typehint import ProcessedLayoutData

logger = logging.getLogger(__name__)


class AddGaussianNoise(nn.Module):
    """
    Add Gaussian Noise to bounding box
    """

    def __init__(
        self,
        mean: float = 0.0,
        std: float = 1.0,
        normalized: bool = True,
        bernoulli_beta: float = 1.0,
    ) -> None:
        super().__init__()

        self.std = std
        self.mean = mean
        self.normalized = normalized
        # adding noise to every element by default
        self.bernoulli_beta = bernoulli_beta
        logger.info(
            "Noise: mean={0}, std={1}, beta={2}".format(
                self.mean, self.std, self.bernoulli_beta
            )
        )

    def __call__(self, data: ProcessedLayoutData) -> ProcessedLayoutData:
        # Gold Label
        if "gold_bboxes" not in data.keys():
            data["gold_bboxes"] = copy.deepcopy(data["bboxes"])

        num_elemnts = data["bboxes"].size(0)
        beta = data["bboxes"].new_ones(num_elemnts) * self.bernoulli_beta
        element_with_noise = torch.bernoulli(beta).unsqueeze(dim=-1)

        if self.normalized:
            data["bboxes"] = (
                data["bboxes"]
                + torch.randn(data["bboxes"].size()) * self.std
                + self.mean
            )
        else:
            canvas_width, canvas_height = data["canvas_size"][0], data["canvas_size"][1]
            ele_x, ele_y = (
                data["bboxes"][:, 0] * canvas_width,
                data["bboxes"][:, 1] * canvas_height,
            )
            ele_w, ele_h = (
                data["bboxes"][:, 2] * canvas_width,
                data["bboxes"][:, 3] * canvas_height,
            )
            data["bboxes"] = torch.stack([ele_x, ele_y, ele_w, ele_h], dim=1)
            data["bboxes"] = (
                data["bboxes"]
                + torch.randn(data["bboxes"].size()) * self.std
                + self.mean
            )
            data["bboxes"][:, 0] /= canvas_width
            data["bboxes"][:, 1] /= canvas_height
            data["bboxes"][:, 2] /= canvas_width
            data["bboxes"][:, 3] /= canvas_height
        data["bboxes"][data["bboxes"] < 0] = 0.0
        data["bboxes"][data["bboxes"] > 1] = 1.0
        data["bboxes"] = data["bboxes"] * element_with_noise + data["gold_bboxes"] * (
            1 - element_with_noise
        )
        return data

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1}, beta={2})".format(
            self.mean, self.std, self.bernoulli_beta
        )
