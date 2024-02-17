import logging
import re
from typing import List, Tuple

import torch
from openai.types.chat import ChatCompletion, ChatCompletionMessage

from layout_prompter.utils import CANVAS_SIZE, ID2LABEL

logger = logging.getLogger(__name__)


class Parser(object):
    def __init__(self, dataset: str, output_format: str):
        self.dataset = dataset
        self.output_format = output_format
        self.id2label = ID2LABEL[self.dataset]
        self.label2id = {v: k for k, v in self.id2label.items()}
        self.canvas_width, self.canvas_height = CANVAS_SIZE[self.dataset]

    def _extract_labels_and_bboxes(
        self, prediction: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.output_format == "seq":
            return self._extract_labels_and_bboxes_from_seq(prediction)
        elif self.output_format == "html":
            return self._extract_labels_and_bboxes_from_html(prediction)
        else:
            raise ValueError(f"Invalid output format: {self.output_format}")

    def _extract_labels_and_bboxes_from_html(
        self, predition: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        labels = re.findall('<div class="(.*?)"', predition)[1:]  # remove the canvas
        x = re.findall(r"left:.?(\d+)px", predition)[1:]
        y = re.findall(r"top:.?(\d+)px", predition)[1:]
        w = re.findall(r"width:.?(\d+)px", predition)[1:]
        h = re.findall(r"height:.?(\d+)px", predition)[1:]
        if not (len(labels) == len(x) == len(y) == len(w) == len(h)):
            raise RuntimeError
        labels_tensor = torch.tensor([self.label2id[label] for label in labels])
        bboxes_tensor = torch.tensor(
            [
                [
                    int(x[i]) / self.canvas_width,
                    int(y[i]) / self.canvas_height,
                    int(w[i]) / self.canvas_width,
                    int(h[i]) / self.canvas_height,
                ]
                for i in range(len(x))
            ]
        )
        return labels_tensor, bboxes_tensor

    def _extract_labels_and_bboxes_from_seq(
        self, prediction: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        label_set = list(self.label2id.keys())
        seq_pattern = r"(" + "|".join(label_set) + r") (\d+) (\d+) (\d+) (\d+)"
        res = re.findall(seq_pattern, prediction)
        labels_tensor = torch.tensor([self.label2id[item[0]] for item in res])
        bboxes_tensor = torch.tensor(
            [
                [
                    int(item[1]) / self.canvas_width,
                    int(item[2]) / self.canvas_height,
                    int(item[3]) / self.canvas_width,
                    int(item[4]) / self.canvas_height,
                ]
                for item in res
            ]
        )
        return labels_tensor, bboxes_tensor

    def __call__(self, response: ChatCompletion):
        assert isinstance(response, ChatCompletion), type(response)

        parsed_predictions: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for choice in response.choices:
            message = choice.message
            assert isinstance(message, ChatCompletionMessage), type(message)
            content = message.content
            assert content is not None
            parsed_predictions.append(self._extract_labels_and_bboxes(content))

        return parsed_predictions
