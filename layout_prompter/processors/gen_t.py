from dataclasses import dataclass
from typing import Tuple

from layout_prompter.processors import TaskProcessor


@dataclass
class GenTypeProcessor(TaskProcessor):
    return_keys: Tuple[str, ...] = (
        "labels",
        "bboxes",
        "gold_bboxes",
        "discrete_bboxes",
        "discrete_gold_bboxes",
    )
    sort_by_pos: bool = False
    shuffle_before_sort_by_label: bool = False
    sort_by_pos_before_sort_by_label: bool = True
