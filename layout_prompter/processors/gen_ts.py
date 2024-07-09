from dataclasses import dataclass
from typing import Tuple

from layout_prompter.processors import TaskProcessor


@dataclass
class GenTypeSizeProcessor(TaskProcessor):
    return_keys: Tuple[str, ...] = (
        "labels",
        "bboxes",
        "gold_bboxes",
        "discrete_bboxes",
        "discrete_gold_bboxes",
    )
    sort_by_pos: bool = False
    shuffle_before_sort_by_label: bool = True
    sort_by_pos_before_sort_by_label: bool = False
