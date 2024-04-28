from dataclasses import dataclass
from typing import Sequence, Tuple


@dataclass
class LayoutDataset(object):
    name: str
    layout_domain: str
    canvas_size: Tuple[int, int]
    labels: Sequence[str]

    def to_index(self, label: str) -> int:
        return self.labels.index(label)

    def to_label(self, index: int) -> str:
        return self.labels[index]

    @property
    def index2label(self) -> dict:
        return {i: label for i, label in enumerate(self.labels)}

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
    labels: Tuple[str, ...] = (
        "text",
        "image",
        "icon",
        "list-item",
        "text-button",
        "toolbar",
        "web-view",
        "input",
        "card",
        "advertisement",
        "background-image",
        "drawer",
        "radio-button",
        "checkbox",
        "multi-tab",
        "pager-indicator",
        "modal",
        "on/off-switch",
        "slider",
        "map-view",
        "button-bar",
        "video",
        "bottom-navigation",
        "number-stepper",
        "date-picker",
    )


@dataclass
class PubLayNetDataset(LayoutDataset):
    name: str = "publaynet"
    layout_domain: str = "document"
    canvas_size: Tuple[int, int] = (120, 160)
    labels: Tuple[str, ...] = (
        "text",
        "title",
        "list",
        "table",
        "figure",
    )


@dataclass
class PosterLayoutDataset(LayoutDataset):
    name: str = "posterlayout"
    layout_domain: str = "poster"
    canvas_size: Tuple[int, int] = (102, 150)
    labels: Tuple[str, ...] = (
        "text",
        "logo",
        "underlay",
    )


@dataclass
class WebUIDataset(LayoutDataset):
    name: str = "webui"
    layout_domain: str = "web"
    canvas_size: Tuple[int, int] = (120, 120)
    labels: Tuple[str, ...] = (
        "text",
        "link",
        "button",
        "title",
        "description",
        "image",
        "background",
        "logo",
        "icon",
        "input",
    )
