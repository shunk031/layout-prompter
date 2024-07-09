from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import seaborn as sns
import torch
from PIL import Image, ImageDraw

from layout_prompter.configs.base import LayoutDatasetConfig
from layout_prompter.modules.rankers import RankerOutput
from layout_prompter.typehint import PilImage

if TYPE_CHECKING:
    from layout_prompter.typehint import ProcessedLayoutData


@dataclass
class VisualizerMixin(object, metaclass=abc.ABCMeta):
    dataset_config: LayoutDatasetConfig
    times: float = 3.0

    @abc.abstractmethod
    def draw_layout(self, *args, **kwargs) -> PilImage:
        raise NotImplementedError

    @abc.abstractmethod
    def __call__(self, predictions: List[RankerOutput]) -> List[PilImage]:
        pass


@dataclass
class Visualizer(VisualizerMixin):
    _colors: Optional[List[Tuple[int, int, int]]] = None

    @property
    def colors(self) -> List[Tuple[int, int, int]]:
        if self._colors is None:
            n_colors = len(self.dataset_config.id2label) + 1
            colors = sns.color_palette("husl", n_colors=n_colors)
            self._colors = [
                (int(c[0] * 255), int(c[1] * 255), int(c[2] * 255)) for c in colors
            ]
        return self._colors

    def draw_layout(
        self, labels_tensor: torch.Tensor, bboxes_tensor: torch.Tensor
    ) -> PilImage:
        canvas_w = self.dataset_config.canvas_width
        canvas_h = self.dataset_config.canvas_height
        img = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))

        draw = ImageDraw.Draw(img, "RGBA")
        labels: List[int] = labels_tensor.tolist()
        bboxes: List[Tuple[float, float, float, float]] = bboxes_tensor.tolist()
        areas = [bbox[2] * bbox[3] for bbox in bboxes]
        indices = sorted(range(len(areas)), key=lambda i: areas[i], reverse=True)

        for i in indices:
            bbox, label = bboxes[i], labels[i]
            color = self.colors[label]
            c_fill = color + (100,)
            x1, y1, x2, y2 = bbox
            x2 += x1
            y2 += y1
            x1, x2 = x1 * canvas_w, x2 * canvas_w
            y1, y2 = y1 * canvas_h, y2 * canvas_h
            draw.rectangle(xy=(x1, y1, x2, y2), outline=color, fill=c_fill)
        return img

    def __call__(
        self, predictions: Union[List[ProcessedLayoutData], List[RankerOutput]]
    ) -> List[PilImage]:
        images: List[PilImage] = []
        for prediction in predictions:
            labels, bboxes = prediction["labels"], prediction["bboxes"]
            img = self.draw_layout(labels, bboxes)
            images.append(img)
        return images
