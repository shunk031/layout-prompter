from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Union

import numpy as np
from PIL import Image, ImageDraw

from layout_prompter.modules.rankers import RankerOutput
from layout_prompter.typehint import PilImage
from layout_prompter.visualizers import VisualizerMixin

if TYPE_CHECKING:
    from layout_prompter.typehint import ProcessedLayoutData


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
            .resize(
                (self.dataset_config.canvas_width, self.dataset_config.canvas_height)
            )
        )
        for prediction in predictions:
            labels, bboxes = prediction["labels"], prediction["bboxes"]
            labels = labels.unsqueeze(-1)
            labels = np.array(labels, dtype=int)
            bboxes = np.array(bboxes)
            bboxes[:, 0::2] *= self.dataset_config.canvas_width
            bboxes[:, 1::2] *= self.dataset_config.canvas_height
            bboxes[:, 2] += bboxes[:, 0]
            bboxes[:, 3] += bboxes[:, 1]
            images.append(
                self.draw_layout(pic, zip(labels, bboxes), zip(labels, bboxes))
            )
        return images
