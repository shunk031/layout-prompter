from dataclasses import dataclass, field
from typing import Dict, Tuple

from .base import LayoutDatasetConfig


@dataclass
class PosterLayoutDatasetConfig(LayoutDatasetConfig):
    name: str = "posterlayout"
    layout_domain: str = "poster"
    canvas_size: Tuple[int, int] = (102, 150)
    id2label: Dict[int, str] = field(
        default_factory=lambda: {
            1: "text",
            2: "logo",
            3: "underlay",
        }
    )
