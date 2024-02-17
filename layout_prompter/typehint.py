from typing import Tuple, TypedDict

import torch


class LayoutData(TypedDict):
    name: str
    bboxes: torch.Tensor
    labels: torch.Tensor
    canvas_size: Tuple[float, float]
    filtered: bool


class Prompt(TypedDict):
    system_prompt: str
    user_prompt: str
