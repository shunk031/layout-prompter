import abc
import logging
from dataclasses import dataclass, field
from typing import Dict, Final, List, Type

import torch

from layout_prompter.transforms import RelationTypes
from layout_prompter.typehint import Prompt
from layout_prompter.utils import CANVAS_SIZE, ID2LABEL, LAYOUT_DOMAIN

logger = logging.getLogger(__name__)

PREAMBLE_TEMPLATE: Final[str] = (
    "Please generate a layout based on the given information. "
    "You need to ensure that the generated layout looks realistic, with elements well aligned and avoiding unnecessary overlap.\n"
    "Task Description: {task_description}\n"
    "Layout Domain: {layout_domain} layout\n"
    "Canvas Size: canvas width is {canvas_width}px, canvas height is {canvas_height}px"
)


HTML_PREFIX: Final[
    str
] = """<html>
<body>
<div class="canvas" style="left: 0px; top: 0px; width: {}px; height: {}px"></div>
"""

HTML_SUFFIX: Final[
    str
] = """</body>
</html>"""

HTML_TEMPLATE: Final[
    str
] = """<div class="{}" style="left: {}px; top: {}px; width: {}px; height: {}px"></div>
"""

HTML_TEMPLATE_WITH_INDEX: Final[
    str
] = """<div class="{}" style="index: {}; left: {}px; top: {}px; width: {}px; height: {}px"></div>
"""


@dataclass
class SerializerMixin(object):
    input_format: str
    output_format: str

    index2label: Dict[int, str]

    canvas_width: int
    canvas_height: int

    task_type: str = "NOT SPECIFIED"

    preamble_template: str = PREAMBLE_TEMPLATE
    add_index_token: bool = True
    add_sep_token: bool = True
    sep_token: str = "|"
    add_unk_token: bool = False
    unk_token: str = "<unk>"

    def _build_seq_input(self, data):
        raise NotImplementedError

    def _build_html_input(self, data):
        raise NotImplementedError

    def build_prompt(self, *args, **kwargs):
        raise NotImplementedError


@dataclass
class Serializer(SerializerMixin, metaclass=abc.ABCMeta):

    def build_input(self, data):
        if self.input_format == "seq":
            return self._build_seq_input(data)
        elif self.input_format == "html":
            return self._build_html_input(data)
        else:
            raise ValueError(f"Unsupported input format: {self.input_format}")

    @abc.abstractmethod
    def _build_seq_input(self, data):
        raise NotImplementedError

    @abc.abstractmethod
    def _build_html_input(self, data):
        raise NotImplementedError

    def build_output(self, data, label_key="labels", bbox_key="discrete_gold_bboxes"):
        if self.output_format == "seq":
            return self._build_seq_output(data, label_key, bbox_key)
        elif self.output_format == "html":
            return self._build_html_output(data, label_key, bbox_key)

    def _build_seq_output(self, data, label_key, bbox_key):
        labels = data[label_key]
        bboxes = data[bbox_key]
        tokens = []

        for idx in range(len(labels)):
            label = self.index2label[int(labels[idx])]
            bbox = bboxes[idx].tolist()
            tokens.append(label)
            if self.add_index_token:
                tokens.append(str(idx))
            tokens.extend(map(str, bbox))
            if self.add_sep_token and idx < len(labels) - 1:
                tokens.append(self.sep_token)
        return " ".join(tokens)

    def _build_html_output(self, data, label_key, bbox_key):
        labels = data[label_key]
        bboxes = data[bbox_key]
        htmls = [HTML_PREFIX.format(self.canvas_width, self.canvas_height)]
        _TEMPLATE = HTML_TEMPLATE_WITH_INDEX if self.add_index_token else HTML_TEMPLATE

        for idx in range(len(labels)):
            label = self.index2label[int(labels[idx])]
            bbox = bboxes[idx].tolist()
            element = [label]
            if self.add_index_token:
                element.append(str(idx))
            element.extend(map(str, bbox))
            htmls.append(_TEMPLATE.format(*element))
        htmls.append(HTML_SUFFIX)
        return "".join(htmls)

    def build_prompt(
        self,
        exemplars: List[Dict[str, torch.Tensor]],
        test_data: Dict[str, torch.Tensor],
        dataset: str,
        max_length: int = 8000,
        separator_in_samples: str = "\n",
        separator_between_samples: str = "\n\n",
    ) -> Prompt:
        system_prompt = self.preamble_template.format(
            task_description=self.task_type,
            layout_domain=LAYOUT_DOMAIN[dataset],
            canvas_width=CANVAS_SIZE[dataset][0],
            canvas_height=CANVAS_SIZE[dataset][1],
        )
        logger.info(f"System prompt: \n{system_prompt}")

        user_prompts: List[str] = []
        for i in range(len(exemplars)):
            _prompt = (
                self.build_input(exemplars[i])
                + separator_in_samples
                + self.build_output(exemplars[i])
            )
            if (
                len(separator_between_samples.join(user_prompts) + _prompt)
                <= max_length
            ):
                user_prompts.append(_prompt)
            else:
                break
        user_prompts.append(self.build_input(test_data) + separator_in_samples)
        user_prompt = separator_between_samples.join(user_prompts)
        logger.info(f"User prompt: \n{user_prompt}")

        return {"system_prompt": system_prompt, "user_prompt": user_prompt}


@dataclass
class GenTypeSerializer(Serializer):
    task_type: str = "generation conditioned on given element types"
    constraint_type: List[str] = field(
        default_factory=lambda: ["Element Type Constraint: "]
    )
    HTML_TEMPLATE_WITHOUT_ANK: str = '<div class="{}"></div>\n'
    HTML_TEMPLATE_WITHOUT_ANK_WITH_INDEX: str = (
        '<div class="{}" style="index: {}"></div>\n'
    )

    def _build_seq_input(self, data: Dict[str, torch.Tensor]) -> str:
        labels = data["labels"]
        tokens = []

        for idx in range(len(labels)):
            label = self.index2label[int(labels[idx])]
            tokens.append(label)
            if self.add_index_token:
                tokens.append(str(idx))
            if self.add_unk_token:
                tokens += [self.unk_token] * 4
            if self.add_sep_token and idx < len(labels) - 1:
                tokens.append(self.sep_token)
        return " ".join(tokens)

    def _build_html_input(self, data) -> str:
        labels = data["labels"]
        htmls = [HTML_PREFIX.format(self.canvas_width, self.canvas_height)]
        if self.add_index_token and self.add_unk_token:
            _TEMPLATE = HTML_TEMPLATE_WITH_INDEX
        elif self.add_index_token and not self.add_unk_token:
            _TEMPLATE = self.HTML_TEMPLATE_WITHOUT_ANK_WITH_INDEX
        elif not self.add_index_token and self.add_unk_token:
            _TEMPLATE = HTML_TEMPLATE
        else:
            _TEMPLATE = self.HTML_TEMPLATE_WITHOUT_ANK

        for idx in range(len(labels)):
            label = self.index2label[int(labels[idx])]
            element = [label]
            if self.add_index_token:
                element.append(str(idx))
            if self.add_unk_token:
                element += [self.unk_token] * 4
            htmls.append(_TEMPLATE.format(*element))
        htmls.append(HTML_SUFFIX)
        return "".join(htmls)

    def build_input(self, data):
        return self.constraint_type[0] + super().build_input(data)


@dataclass
class GenTypeSizeSerializer(Serializer):
    task_type: str = "generation conditioned on given element types and sizes"
    constraint_type: List[str] = field(
        default_factory=lambda: ["Element Type and Size Constraint: "]
    )
    HTML_TEMPLATE_WITHOUT_ANK: str = (
        '<div class="{}" style="width: {}px; height: {}px"></div>\n'
    )
    HTML_TEMPLATE_WITHOUT_ANK_WITH_INDEX: str = (
        '<div class="{}" style="index: {}; width: {}px; height: {}px"></div>\n'
    )

    def _build_seq_input(self, data):
        labels = data["labels"]
        bboxes = data["discrete_gold_bboxes"]
        tokens = []

        for idx in range(len(labels)):
            label = self.index2label[int(labels[idx])]
            bbox = bboxes[idx].tolist()
            tokens.append(label)
            if self.add_index_token:
                tokens.append(str(idx))
            if self.add_unk_token:
                tokens += [self.unk_token] * 2
            tokens.extend(map(str, bbox[2:]))
            if self.add_sep_token and idx < len(labels) - 1:
                tokens.append(self.sep_token)
        return " ".join(tokens)

    def _build_html_input(self, data):
        labels = data["labels"]
        bboxes = data["discrete_gold_bboxes"]
        htmls = [HTML_PREFIX.format(self.canvas_width, self.canvas_height)]
        if self.add_index_token and self.add_unk_token:
            _TEMPLATE = HTML_TEMPLATE_WITH_INDEX
        elif self.add_index_token and not self.add_unk_token:
            _TEMPLATE = self.HTML_TEMPLATE_WITHOUT_ANK_WITH_INDEX
        elif not self.add_index_token and self.add_unk_token:
            _TEMPLATE = HTML_TEMPLATE
        else:
            _TEMPLATE = self.HTML_TEMPLATE_WITHOUT_ANK

        for idx in range(len(labels)):
            label = self.index2label[int(labels[idx])]
            bbox = bboxes[idx].tolist()
            element = [label]
            if self.add_index_token:
                element.append(str(idx))
            if self.add_unk_token:
                element += [self.unk_token] * 2
            element.extend(map(str, bbox[2:]))
            htmls.append(_TEMPLATE.format(*element))
        htmls.append(HTML_SUFFIX)
        return "".join(htmls)

    def build_input(self, data):
        return self.constraint_type[0] + super().build_input(data)


@dataclass
class GenRelationSerializer(Serializer):
    task_type: str = (
        "generation conditioned on given element relationships\n"
        "'A left B' means that the center coordinate of A is to the left of the center coordinate of B. "
        "'A right B' means that the center coordinate of A is to the right of the center coordinate of B. "
        "'A top B' means that the center coordinate of A is above the center coordinate of B. "
        "'A bottom B' means that the center coordinate of A is below the center coordinate of B. "
        "'A center B' means that the center coordinate of A and the center coordinate of B are very close. "
        "'A smaller B' means that the area of A is smaller than the ares of B. "
        "'A larger B' means that the area of A is larger than the ares of B. "
        "'A equal B' means that the area of A and the ares of B are very close. "
        "Here, center coordinate = (left + width / 2, top + height / 2), "
        "area = width * height"
    )
    constraint_type: List[str] = field(
        default_factory=lambda: [
            "Element Type Constraint: ",
            "Element Relationship Constraint: ",
        ]
    )
    HTML_TEMPLATE_WITHOUT_ANK: str = '<div class="{}"></div>\n'
    HTML_TEMPLATE_WITHOUT_ANK_WITH_INDEX: str = (
        '<div class="{}" style="index: {}"></div>\n'
    )
    index2type: Dict[int, str] = field(
        default_factory=lambda: RelationTypes.index2type()
    )

    def _build_seq_input(self, data):
        labels = data["labels"]
        relations = data["relations"]
        tokens = []

        for idx in range(len(labels)):
            label = self.index2label[int(labels[idx])]
            tokens.append(label)
            if self.add_index_token:
                tokens.append(str(idx))
            if self.add_unk_token:
                tokens += [self.unk_token] * 4
            if self.add_sep_token and idx < len(labels) - 1:
                tokens.append(self.sep_token)
        type_cons = " ".join(tokens)
        if len(relations) == 0:
            return self.constraint_type[0] + type_cons
        tokens = []
        for idx in range(len(relations)):
            label_i = relations[idx][2]
            index_i = relations[idx][3]
            if label_i != 0:
                tokens.append("{} {}".format(self.index2label[int(label_i)], index_i))
            else:
                tokens.append("canvas")
            tokens.append(self.index2type[int(relations[idx][4])])
            label_j = relations[idx][0]
            index_j = relations[idx][1]
            if label_j != 0:
                tokens.append("{} {}".format(self.index2label[int(label_j)], index_j))
            else:
                tokens.append("canvas")
            if self.add_sep_token and idx < len(relations) - 1:
                tokens.append(self.sep_token)
        relation_cons = " ".join(tokens)
        return (
            self.constraint_type[0]
            + type_cons
            + "\n"
            + self.constraint_type[1]
            + relation_cons
        )

    def _build_html_input(self, data):
        labels = data["labels"]
        relations = data["relations"]
        htmls = [HTML_PREFIX.format(self.canvas_width, self.canvas_height)]
        if self.add_index_token and self.add_unk_token:
            _TEMPLATE = HTML_TEMPLATE_WITH_INDEX
        elif self.add_index_token and not self.add_unk_token:
            _TEMPLATE = self.HTML_TEMPLATE_WITHOUT_ANK_WITH_INDEX
        elif not self.add_index_token and self.add_unk_token:
            _TEMPLATE = HTML_TEMPLATE
        else:
            _TEMPLATE = self.HTML_TEMPLATE_WITHOUT_ANK

        for idx in range(len(labels)):
            label = self.index2label[int(labels[idx])]
            element = [label]
            if self.add_index_token:
                element.append(str(idx))
            if self.add_unk_token:
                element += [self.unk_token] * 4
            htmls.append(_TEMPLATE.format(*element))
        htmls.append(HTML_SUFFIX)
        type_cons = "".join(htmls)
        if len(relations) == 0:
            return self.constraint_type[0] + type_cons
        tokens = []
        for idx in range(len(relations)):
            label_i = relations[idx][2]
            index_i = relations[idx][3]
            if label_i != 0:
                tokens.append("{} {}".format(self.index2label[int(label_i)], index_i))
            else:
                tokens.append("canvas")
            tokens.append(self.index2type[int(relations[idx][4])])
            label_j = relations[idx][0]
            index_j = relations[idx][1]
            if label_j != 0:
                tokens.append("{} {}".format(self.index2label[int(label_j)], index_j))
            else:
                tokens.append("canvas")
            if self.add_sep_token and idx < len(relations) - 1:
                tokens.append(self.sep_token)
        relation_cons = " ".join(tokens)
        return (
            self.constraint_type[0]
            + type_cons
            + "\n"
            + self.constraint_type[1]
            + relation_cons
        )


@dataclass
class CompletionSerializer(Serializer):
    task_type: str = "layout completion"
    constraint_type: List[str] = field(default_factory=lambda: ["Partial Layout: "])

    def _build_seq_input(self, data):
        data["partial_labels"] = data["labels"][:1]
        data["partial_bboxes"] = data["discrete_bboxes"][:1, :]
        return self._build_seq_output(data, "partial_labels", "partial_bboxes")

    def _build_html_input(self, data):
        data["partial_labels"] = data["labels"][:1]
        data["partial_bboxes"] = data["discrete_bboxes"][:1, :]
        return self._build_html_output(data, "partial_labels", "partial_bboxes")

    def build_input(self, data):
        return self.constraint_type[0] + super().build_input(data)


@dataclass
class RefinementSerializer(Serializer):
    task_type: str = "layout refinement"
    constraint_type: List[str] = field(default_factory=lambda: ["Noise Layout: "])

    def _build_seq_input(self, data):
        return self._build_seq_output(data, "labels", "discrete_bboxes")

    def _build_html_input(self, data):
        return self._build_html_output(data, "labels", "discrete_bboxes")

    def build_input(self, data):
        return self.constraint_type[0] + super().build_input(data)


@dataclass
class ContentAwareSerializer(Serializer):
    task_type: str = (
        "content-aware layout generation\n"
        "Please place the following elements to avoid salient content, and underlay must be the background of text or logo."
    )
    constraint_type: List[str] = field(
        default_factory=lambda: ["Content Constraint: ", "Element Type Constraint: "]
    )
    CONTENT_TEMPLATE: str = "left {}px, top {}px, width {}px, height {}px"

    def _build_seq_input(self, data):
        labels = data["labels"]
        content_bboxes = data["discrete_content_bboxes"]

        tokens = []
        for idx in range(len(content_bboxes)):
            content_bbox = content_bboxes[idx].tolist()
            tokens.append(self.CONTENT_TEMPLATE.format(*content_bbox))
            if self.add_index_token and idx < len(content_bboxes) - 1:
                tokens.append(self.sep_token)
        content_cons = " ".join(tokens)

        tokens = []
        for idx in range(len(labels)):
            label = self.index2label[int(labels[idx])]
            tokens.append(label)
            if self.add_index_token:
                tokens.append(str(idx))
            if self.add_unk_token:
                tokens += [self.unk_token] * 4
            if self.add_sep_token and idx < len(labels) - 1:
                tokens.append(self.sep_token)
        type_cons = " ".join(tokens)
        return (
            self.constraint_type[0]
            + content_cons
            + "\n"
            + self.constraint_type[1]
            + type_cons
        )


@dataclass
class TextToLayoutSerializer(Serializer):
    task_type: str = (
        "text-to-layout\n"
        "There are ten optional element types, including: image, icon, logo, background, title, description, text, link, input, button. "
        "Please do not exceed the boundaries of the canvas. "
        "Besides, do not generate elements at the edge of the canvas, that is, reduce top: 0px and left: 0px predictions as much as possible."
    )
    constraint_type: List[str] = field(default_factory=lambda: ["Text: "])

    def _build_seq_input(self, data):
        return data["text"]

    def build_input(self, data):
        return self.constraint_type[0] + super().build_input(data)


SERIALIZER_MAP: Dict[str, Type[SerializerMixin]] = {
    "gen-t": GenTypeSerializer,
    "gen-ts": GenTypeSizeSerializer,
    "gen-r": GenRelationSerializer,
    "completion": CompletionSerializer,
    "refinement": RefinementSerializer,
    "content": ContentAwareSerializer,
    "text": TextToLayoutSerializer,
}


def create_serializer(
    dataset: str,
    task: str,
    input_format: str,
    output_format: str,
    add_index_token: bool,
    add_sep_token: bool,
    add_unk_token: bool,
) -> SerializerMixin:
    serializer_cls = SERIALIZER_MAP[task]
    index2label = ID2LABEL[dataset]
    canvas_width, canvas_height = CANVAS_SIZE[dataset]
    serializer = serializer_cls(
        input_format=input_format,
        output_format=output_format,
        index2label=index2label,
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        add_index_token=add_index_token,
        add_sep_token=add_sep_token,
        add_unk_token=add_unk_token,
    )
    return serializer
