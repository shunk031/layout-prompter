import base64
import io
import os
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

import torch
from PIL import Image

from layout_prompter.processors import TaskProcessor
from layout_prompter.transforms import SaliencyMapToBBoxes
from layout_prompter.typehint import PilImage

CONTENT_IMAGE_FORMAT: Literal["png"] = "png"


@dataclass
class ContentAwareProcessor(TaskProcessor):
    return_keys: Tuple[str, ...] = (
        "idx",
        "labels",
        "bboxes",
        "gold_bboxes",
        "content_bboxes",
        "discrete_bboxes",
        "discrete_gold_bboxes",
        "discrete_content_bboxes",
        "inpainted_image",
    )

    sort_by_pos: bool = False
    shuffle_before_sort_by_label: bool = False
    sort_by_pos_before_sort_by_label: bool = True
    filter_threshold: int = 100
    max_element_numbers: int = 10

    possible_labels: List[torch.Tensor] = field(default_factory=list)

    @property
    def saliency_map_to_bboxes(self) -> SaliencyMapToBBoxes:
        return SaliencyMapToBBoxes(threshold=self.filter_threshold)

    def _encode_image(self, image: PilImage) -> str:
        image = image.convert("RGB")
        image_byte = io.BytesIO()
        image.save(image_byte, format=CONTENT_IMAGE_FORMAT)
        return base64.b64encode(image_byte.getvalue()).decode("utf-8")

    def _normalize_bboxes(self, bboxes, w: int, h: int):
        bboxes = bboxes.float()
        bboxes[:, 0::2] /= w
        bboxes[:, 1::2] /= h
        return bboxes

    def __call__(  # type: ignore[override]
        self,
        idx: int,
        split: str,
        saliency_map_path: os.PathLike,
        inpainted_image_path: Optional[os.PathLike] = None,
    ) -> Optional[Dict[str, torch.Tensor]]:
        saliency_map = Image.open(saliency_map_path)  # type: ignore
        content_bboxes = self.saliency_map_to_bboxes(saliency_map)
        if len(content_bboxes) == 0:
            return None

        map_w, map_h = saliency_map.size
        content_bboxes = self._normalize_bboxes(content_bboxes, w=map_w, h=map_h)

        encoded_inpainted_image: Optional[str] = None
        if inpainted_image_path is not None:
            inpainted_image = Image.open(inpainted_image_path)  # type: ignore
            assert inpainted_image.size == saliency_map.size

            encoded_inpainted_image = self._encode_image(inpainted_image)

        if split == "train":
            assert self.metadata is not None
            _metadata = self.metadata[
                (self.metadata["poster_path"] == f"train/{idx}.png")
                & (self.metadata["cls_elem"] > 0)
            ]
            labels = torch.tensor(list(map(int, _metadata["cls_elem"])))
            bboxes = torch.tensor(list(map(eval, _metadata["box_elem"])))
            if len(labels) == 0:
                return None

            bboxes[:, 2] -= bboxes[:, 0]
            bboxes[:, 3] -= bboxes[:, 1]
            bboxes = self._normalize_bboxes(bboxes, w=map_w, h=map_h)
            if len(labels) <= self.max_element_numbers:
                self.possible_labels.append(labels)

            data = {
                "idx": idx,
                "labels": labels,
                "bboxes": bboxes,
                "content_bboxes": content_bboxes,
                "inpainted_image": encoded_inpainted_image,
            }
        else:
            if len(self.possible_labels) == 0:
                raise RuntimeError("Please process training data first")

            breakpoint()
            labels = random.choice(self.possible_labels)
            data = {
                "idx": idx,
                "labels": labels,
                "bboxes": torch.zeros((len(labels), 4)),  # dummy
                "content_bboxes": content_bboxes,
                "inpainted_image": encoded_inpainted_image,
            }

        return super().__call__(data)  # type: ignore
