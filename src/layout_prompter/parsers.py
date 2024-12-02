import logging
import re
import textwrap

import openai
import torch

from layout_prompter.exception import LayoutPrompterException
from layout_prompter.utils import CANVAS_SIZE, ID2LABEL

logger = logging.getLogger(__name__)


class Parser:
    def __init__(self, dataset: str, output_format: str):
        self.dataset = dataset
        self.output_format = output_format
        self.id2label = ID2LABEL[self.dataset]
        self.label2id = {v: k for k, v in self.id2label.items()}
        self.canvas_width, self.canvas_height = CANVAS_SIZE[self.dataset]

    def _extract_labels_and_bboxes(self, prediction: str):
        if self.output_format == "seq":
            return self._extract_labels_and_bboxes_from_seq(prediction)
        elif self.output_format == "html":
            return self._extract_labels_and_bboxes_from_html(prediction)

    def _extract_labels_and_bboxes_from_html(self, prediction: str):
        logger.debug(f"Prediction result:\n{prediction}")

        labels = re.findall('<div class="(.*?)"', prediction)
        x = re.findall(r"left:.?(\d+)px", prediction)
        y = re.findall(r"top:.?(\d+)px", prediction)
        w = re.findall(r"width:.?(\d+)px", prediction)
        h = re.findall(r"height:.?(\d+)px", prediction)

        num_elements = (
            f"labels={len(labels)}, x={len(x)}, y={len(y)}, w={len(w)}, h={len(h)}"
        )
        logger.debug(f"# parsed elements: {num_elements}")

        if len(labels[1:]) == len(x[1:]) == len(y[1:]) == len(w[1:]) == len(h[1:]):
            # Remove the canvas-related information from all the elements
            labels, x, y, w, h = labels[1:], x[1:], y[1:], w[1:], h[1:]
        elif len(labels[1:]) == len(x) == len(y) == len(w[1:]) == len(h[1:]):
            # This is executed when only the canvas contains width and height
            # Remove the canvas-related information from the labels and the width and height
            labels, w, h = labels[1:], w[1:], h[1:]
        else:
            msg = textwrap.dedent(
                f"""\
                The number of {num_elements} are not the same.
                Details:
                    {labels=}
                    {x=}
                    {y=}
                    {w=}
                    {h=}
                """
            )
            raise LayoutPrompterException(msg)

        # Ensure that the number of labels, x, y, w, and h are the same
        assert len(labels) == len(x) == len(y) == len(w) == len(h)

        if len(labels) < 1:
            raise LayoutPrompterException(
                "No labels and bboxes found in the prediction"
            )
        try:
            labels = torch.tensor([self.label2id[label] for label in labels])
        except KeyError as err:
            raise LayoutPrompterException(
                f"Label not found in the mapping (= {self.label2id})"
            ) from err

        bboxes = torch.tensor(
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
        return labels, bboxes

    def _extract_labels_and_bboxes_from_seq(self, prediction: str):
        label_set = list(self.label2id.keys())
        seq_pattern = r"(" + "|".join(label_set) + r") (\d+) (\d+) (\d+) (\d+)"
        res = re.findall(seq_pattern, prediction)
        labels = torch.tensor([self.label2id[item[0]] for item in res])
        bboxes = torch.tensor(
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
        return labels, bboxes

    def __call__(self, predictions):
        parsed_predictions = []
        for choice in predictions.choices:
            message = choice.message
            content = message.content
            try:
                parsed_predictions.append(self._extract_labels_and_bboxes(content))
            except LayoutPrompterException as err:
                logger.warning(err, exc_info=True)
                continue

        return parsed_predictions
