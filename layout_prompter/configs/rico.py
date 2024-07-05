from dataclasses import dataclass, field
from typing import Dict, Tuple

from .base import LayoutDatasetConfig


@dataclass
class RicoDatasetConfig(LayoutDatasetConfig):
    name: str = "rico"
    layout_domain: str = "android"
    canvas_size: Tuple[int, int] = (90, 160)
    id2label: Dict[int, str] = field(
        default_factory=lambda: {
            1: "text",
            2: "image",
            3: "icon",
            4: "list-item",
            5: "text-button",
            6: "toolbar",
            7: "web-view",
            8: "input",
            9: "card",
            10: "advertisement",
            11: "background-image",
            12: "drawer",
            13: "radio-button",
            14: "checkbox",
            15: "multi-tab",
            16: "pager-indicator",
            17: "modal",
            18: "on/off-switch",
            19: "slider",
            20: "map-view",
            21: "button-bar",
            22: "video",
            23: "bottom-navigation",
            24: "number-stepper",
            25: "date-picker",
        }
    )
