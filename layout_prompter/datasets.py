from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


@dataclass
class LayoutDataset(object):
    name: str
    layout_domain: str
    canvas_size: Tuple[int, int]

    id2label: Dict[int, str]
    _label2id: Optional[Dict[str, int]] = None

    def __post_init__(self) -> None:
        self._label2id = {v: k for k, v in self.id2label.items()}

    @property
    def label2id(self) -> Dict[str, int]:
        assert self._label2id is not None
        return self._label2id

    @property
    def canvas_width(self) -> int:
        width, _ = self.canvas_size
        return width

    @property
    def canvas_height(self) -> int:
        _, height = self.canvas_size
        return height


@dataclass
class RicoDataset(LayoutDataset):
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


@dataclass
class PubLayNetDataset(LayoutDataset):
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


@dataclass
class PosterLayoutDataset(LayoutDataset):
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


@dataclass
class WebUIDataset(LayoutDataset):
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
