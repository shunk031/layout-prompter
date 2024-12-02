from typing import Annotated, Any, Dict, List, Literal, Tuple, TypedDict, Union

import torch
from PIL import Image

JsonDict = Dict[str, Any]

PilImage = Annotated[Image.Image, "PIL Image"]

InOutFormat = Literal["seq", "html"]

ConstraintExplicitTask = Literal[
    "gen-t",
    "gen-ts",
    "gen-r",
    "completion",
    "refinement",
]
ContentAwareTask = Literal["content-aware"]
TextToLayoutTask = Literal["text-to-layout"]
Task = Union[ConstraintExplicitTask, ContentAwareTask, TextToLayoutTask]

ContentAgnosticDataset = Literal["rico", "publaynet"]
ContentAwareDataset = Literal["posterlayout"]
TextToLayoutDataset = Literal["webui"]


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
