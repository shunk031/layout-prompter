from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class LayoutDatasetConfig(object):
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
