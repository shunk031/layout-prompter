from dataclasses import dataclass, field
from typing import Dict, Tuple

from .base import LayoutDatasetConfig


@dataclass
class WebUIDatasetConfig(LayoutDatasetConfig):
    name: str = "webui"
    layout_domain: str = "web"
    canvas_size: Tuple[int, int] = (120, 120)
    id2label: Dict[int, str] = field(
        default_factory=lambda: {
            0: "text",
            1: "link",
            2: "button",
            3: "title",
            4: "description",
            5: "image",
            6: "background",
            7: "logo",
            8: "icon",
            9: "input",
        }
    )
