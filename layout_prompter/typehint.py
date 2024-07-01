from typing import Annotated, Any, Dict, List, Literal, Tuple, TypedDict

import torch
from PIL.Image import Image

JsonDict = Dict[str, Any]

Task = Literal[
    "gen-t", "gen-ts", "gen-r", "completion", "refinement", "content", "text"
]

InOutFormat = Literal["seq", "html"]

PilImage = Annotated[Image, "Pillow Image"]


class LayoutData(TypedDict):
    name: str
    bboxes: torch.Tensor
    labels: torch.Tensor
    canvas_size: Tuple[float, float]
    filtered: bool


class TextToLayoutData(TypedDict):
    text: str
    canvas_width: int
    elements: List[JsonDict]


class ProcessedLayoutData(TypedDict):
    name: str
    bboxes: torch.Tensor
    labels: torch.Tensor
    gold_bboxes: torch.Tensor
    discrete_bboxes: torch.Tensor
    discrete_gold_bboxes: torch.Tensor

    content_bboxes: torch.Tensor
    discrete_content_bboxes: torch.Tensor

    canvas_size: Tuple[float, float]

    ori_bboxes: torch.Tensor
    ori_labels: torch.Tensor

    embedding: torch.Tensor


class Prompt(TypedDict):
    system_prompt: str
    user_prompt: str
