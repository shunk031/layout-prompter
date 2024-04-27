from __future__ import annotations

import abc
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import numpy as np
import seaborn as sns
import torch
from PIL import Image, ImageDraw
from PIL.Image import Image as PilImage

from layout_prompter.datasets import LayoutDataset
from layout_prompter.modules.rankers import RankerOutput

if TYPE_CHECKING:
    from layout_prompter.typehint import ProcessedLayoutData


@dataclass
class VisualizerMixin(object, metaclass=abc.ABCMeta):
    dataset: LayoutDataset
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
            n_colors = len(self.dataset.id2label) + 1
            colors = sns.color_palette("husl", n_colors=n_colors)
            self._colors = [
                (int(c[0] * 255), int(c[1] * 255), int(c[2] * 255)) for c in colors
            ]
        return self._colors

    def draw_layout(
        self, labels_tensor: torch.Tensor, bboxes_tensor: torch.Tensor
    ) -> PilImage:
        canvas_w = self.dataset.canvas_width
        canvas_h = self.dataset.canvas_height
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


@dataclass
class ContentAwareVisualizer(VisualizerMixin):
    canvas_path: str = ""

    def __post_init__(self) -> None:
        assert self.canvas_path != "", "`canvas_path` is required."

    def draw_layout(self, img, elems, elems2):
        drawn_outline = img.copy()
        drawn_fill = img.copy()
        draw_ol = ImageDraw.ImageDraw(drawn_outline)
        draw_f = ImageDraw.ImageDraw(drawn_fill)

        cls_color_dict = {1: "green", 2: "red", 3: "orange"}

        for cls, box in elems:
            if cls[0]:
                draw_ol.rectangle(
                    tuple(box), fill=None, outline=cls_color_dict[cls[0]], width=5
                )

        s_elems = sorted(list(elems2), key=lambda x: x[0], reverse=True)
        for cls, box in s_elems:
            if cls[0]:
                draw_f.rectangle(tuple(box), fill=cls_color_dict[cls[0]])

        drawn_outline = drawn_outline.convert("RGBA")
        drawn_fill = drawn_fill.convert("RGBA")
        drawn_fill.putalpha(int(256 * 0.3))
        drawn = Image.alpha_composite(drawn_outline, drawn_fill)

        return drawn

    def __call__(  # type: ignore[override]
        self,
        predictions: Union[List[ProcessedLayoutData], List[RankerOutput]],
        test_idx: int,
    ) -> List[PilImage]:
        images = []
        pic = (
            Image.open(os.path.join(self.canvas_path, f"{test_idx}.png"))
            .convert("RGB")
            .resize((self.dataset.canvas_width, self.dataset.canvas_height))
        )
        for prediction in predictions:
            labels, bboxes = prediction["labels"], prediction["bboxes"]
            labels = labels.unsqueeze(-1)
            labels = np.array(labels, dtype=int)
            bboxes = np.array(bboxes)
            bboxes[:, 0::2] *= self.dataset.canvas_width
            bboxes[:, 1::2] *= self.dataset.canvas_height
            bboxes[:, 2] += bboxes[:, 0]
            bboxes[:, 3] += bboxes[:, 1]
            images.append(
                self.draw_layout(pic, zip(labels, bboxes), zip(labels, bboxes))
            )
        return images


def create_image_grid(
    image_list: List[PilImage],
    rows: int = 2,
    cols: int = 5,
    border_size: int = 6,
    border_color: Tuple[int, int, int] = (0, 0, 0),
) -> PilImage:
    result_width = (
        image_list[0].width * cols + (cols - 1) * border_size + 2 * border_size
    )
    result_height = (
        image_list[0].height * rows + (rows - 1) * border_size + 2 * border_size
    )
    result_image = Image.new("RGB", (result_width, result_height), border_color)
    draw = ImageDraw.Draw(result_image)

    outer_border_rect = [0, 0, result_width, result_height]
    draw.rectangle(outer_border_rect, outline=border_color, width=border_size)

    for i in range(len(image_list)):
        row = i // cols
        col = i % cols
        x_offset = col * (image_list[i].width + border_size) + border_size
        y_offset = row * (image_list[i].height + border_size) + border_size
        result_image.paste(image_list[i], (x_offset, y_offset))

        if border_size > 0:
            border_rect = [
                x_offset - border_size,
                y_offset - border_size,
                x_offset + image_list[i].width + border_size,
                y_offset + image_list[i].height + border_size,
            ]
            draw.rectangle(border_rect, outline=border_color, width=border_size)

    return result_image
