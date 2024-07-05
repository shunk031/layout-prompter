import abc
import logging
import re
from dataclasses import dataclass
from typing import List, TypedDict

import torch

from layout_prompter.configs import LayoutDatasetConfig

logger = logging.getLogger(__name__)


class ParserOutput(TypedDict):
    bboxes: torch.Tensor
    labels: torch.Tensor


@dataclass
class Parser(object, metaclass=abc.ABCMeta):
    dataset_config: LayoutDatasetConfig
    output_format: str

    def _extract_labels_and_bboxes(self, prediction: str) -> ParserOutput:
        if self.output_format == "seq":
            return self._extract_labels_and_bboxes_from_seq(prediction)
        elif self.output_format == "html":
            return self._extract_labels_and_bboxes_from_html(prediction)
        else:
            raise ValueError(f"Invalid output format: {self.output_format}")

    def _extract_labels_and_bboxes_from_html(self, predition: str) -> ParserOutput:
        labels = re.findall('<div class="(.*?)"', predition)[1:]  # remove the canvas
        x = re.findall(r"left:.?(\d+)px", predition)[1:]
        y = re.findall(r"top:.?(\d+)px", predition)[1:]
        w = re.findall(r"width:.?(\d+)px", predition)[1:]
        h = re.findall(r"height:.?(\d+)px", predition)[1:]

        if not (len(labels) == len(x) == len(y) == len(w) == len(h)):
            raise RuntimeError(
                "The number of labels, x, y, w, h are not the same "
                f"(#labels = {len(labels)}, #x = {len(x)}, #y = {len(y)}, #w = {len(w)}, #h = {len(h)})."
            )

        labels_tensor = torch.tensor(
            [self.dataset_config.label2id[label] for label in labels]
        )
        bboxes_tensor = torch.tensor(
            [
                [
                    int(x[i]) / self.dataset_config.canvas_width,
                    int(y[i]) / self.dataset_config.canvas_height,
                    int(w[i]) / self.dataset_config.canvas_width,
                    int(h[i]) / self.dataset_config.canvas_height,
                ]
                for i in range(len(x))
            ]
        )
        return {"bboxes": bboxes_tensor, "labels": labels_tensor}

    def _extract_labels_and_bboxes_from_seq(self, prediction: str) -> ParserOutput:
        label_set = list(self.dataset_config.label2id.keys())
        seq_pattern = r"(" + "|".join(label_set) + r") (\d+) (\d+) (\d+) (\d+)"
        res = re.findall(seq_pattern, prediction)
        labels_tensor = torch.tensor(
            [self.dataset_config.label2id[item[0]] for item in res]
        )
        bboxes_tensor = torch.tensor(
            [
                [
                    int(item[1]) / self.dataset_config.canvas_width,
                    int(item[2]) / self.dataset_config.canvas_height,
                    int(item[3]) / self.dataset_config.canvas_width,
                    int(item[4]) / self.dataset_config.canvas_height,
                ]
                for item in res
            ]
        )
        return {"bboxes": bboxes_tensor, "labels": labels_tensor}

    @abc.abstractmethod
    def parse(self, response, *args, **kwargs) -> List[ParserOutput]:
        raise NotImplementedError

    def log_filter_response_count(
        self, num_return: int, parsed_response: List[ParserOutput]
    ) -> None:
        logger.debug(f"Filter {num_return - len(parsed_response)} invalid response.")

    def check_filtered_response_count(
        self, original_response, parsed_response: List[ParserOutput]
    ) -> None:
        pass

    def __call__(self, response, *args, **kwargs) -> List[ParserOutput]:
        parsed_response = self.parse(response, *args, **kwargs)
        self.check_filtered_response_count(response, parsed_response)

        return parsed_response
