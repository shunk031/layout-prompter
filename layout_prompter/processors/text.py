import copy
from dataclasses import dataclass
from typing import Tuple, TypedDict

import torch

from layout_prompter.processors import TaskProcessorMixin
from layout_prompter.transforms import CLIPTextEncoderTransform
from layout_prompter.typehint import TextToLayoutData
from layout_prompter.utils import clean_text


class TextToLayoutProcessorOutput(TypedDict):
    text: str
    embedding: torch.Tensor
    labels: torch.Tensor
    discrete_gold_bboxes: torch.Tensor
    discrete_bboxes: torch.Tensor


@dataclass
class TextToLayoutProcessor(TaskProcessorMixin):
    return_keys: Tuple[str, ...] = (
        "labels",
        "bboxes",
        "text",
        "embedding",
    )
    text_encode_transform: CLIPTextEncoderTransform = CLIPTextEncoderTransform()

    def _scale(self, original_width, elements_):
        elements = copy.deepcopy(elements_)
        ratio = self.dataset_config.canvas_width / original_width
        for i in range(len(elements)):
            elements[i]["position"][0] = int(ratio * elements[i]["position"][0])
            elements[i]["position"][1] = int(ratio * elements[i]["position"][1])
            elements[i]["position"][2] = int(ratio * elements[i]["position"][2])
            elements[i]["position"][3] = int(ratio * elements[i]["position"][3])
        return elements

    def __call__(  # type: ignore[override]
        self,
        data: TextToLayoutData,
    ) -> TextToLayoutProcessorOutput:
        text = clean_text(data["text"])

        embedding = self.text_encode_transform(
            clean_text(data["text"], remove_summary=True)
        )
        original_width = data["canvas_width"]
        elements = data["elements"]
        elements = self._scale(original_width, elements)
        elements = sorted(elements, key=lambda x: (x["position"][1], x["position"][0]))

        labels = [self.dataset_config.label2id[element["type"]] for element in elements]
        labels_tensor = torch.tensor(labels)
        bboxes = [element["position"] for element in elements]
        bboxes_tensor = torch.tensor(bboxes)

        return {
            "text": text,
            "embedding": embedding,
            "labels": labels_tensor,
            "discrete_gold_bboxes": bboxes_tensor,
            "discrete_bboxes": bboxes_tensor,
        }
