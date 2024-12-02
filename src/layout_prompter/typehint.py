from typing import Annotated, Any, Dict, Literal, Union

from PIL import Image

JsonDict = Dict[str, Any]

PilImage = Annotated[Image.Image, "PIL Image"]

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
