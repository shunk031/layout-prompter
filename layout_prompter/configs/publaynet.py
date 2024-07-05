from dataclasses import dataclass, field
from typing import Dict, Tuple

from .base import LayoutDatasetConfig


@dataclass
class PubLayNetDatasetConfig(LayoutDatasetConfig):
    name: str = "publaynet"
    layout_domain: str = "document"
    canvas_size: Tuple[int, int] = (120, 160)
    id2label: Dict[int, str] = field(
        default_factory=lambda: {
            1: "text",
            2: "title",
            3: "list",
            4: "table",
            5: "figure",
        }
    )
