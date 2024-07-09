from dataclasses import dataclass
from typing import Tuple

from layout_prompter.processors import TaskProcessor
from layout_prompter.transforms import (
    AddCanvasElement,
    AddRelation,
    DiscretizeBoundingBox,
)


@dataclass
class GenRelationProcessor(TaskProcessor):
    return_keys: Tuple[str, ...] = (
        "labels",
        "bboxes",
        "gold_bboxes",
        "discrete_bboxes",
        "discrete_gold_bboxes",
        "relations",
    )
    sort_by_pos: bool = False
    shuffle_before_sort_by_label: bool = False
    sort_by_pos_before_sort_by_label: bool = True
    relation_constrained_discrete_before_induce_relations: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        assert self.transform_functions is not None

        self.transform_functions = self.transform_functions[:-1]
        if self.relation_constrained_discrete_before_induce_relations:
            self.transform_functions.append(
                DiscretizeBoundingBox(
                    num_x_grid=self.dataset_config.canvas_width,
                    num_y_grid=self.dataset_config.canvas_height,
                )
            )
            self.transform_functions.append(
                AddCanvasElement(discrete_fn=self.transform_functions[-1])
            )
            self.transform_functions.append(AddRelation())
        else:
            self.transform_functions.append(AddCanvasElement())
            self.transform_functions.append(AddRelation())
            self.transform_functions.append(
                DiscretizeBoundingBox(
                    num_x_grid=self.dataset_config.canvas_width,
                    num_y_grid=self.dataset_config.canvas_height,
                )
            )
