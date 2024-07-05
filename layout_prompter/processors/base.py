import copy
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch.nn as nn
import torchvision.transforms as T

from layout_prompter.configs import LayoutDatasetConfig
from layout_prompter.transforms import (
    DiscretizeBoundingBox,
    LabelDictSort,
    LexicographicSort,
    ShuffleElements,
)
from layout_prompter.typehint import LayoutData, ProcessedLayoutData


@dataclass
class TaskProcessorMixin(object):
    dataset_config: LayoutDatasetConfig
    return_keys: Optional[Tuple[str, ...]] = None

    def __post_init__(self) -> None:
        assert self.return_keys is not None

    def __call__(self, data: LayoutData) -> ProcessedLayoutData:
        raise NotImplementedError


@dataclass
class TaskProcessor(TaskProcessorMixin):
    sort_by_pos: Optional[bool] = None
    shuffle_before_sort_by_label: Optional[bool] = None
    sort_by_pos_before_sort_by_label: Optional[bool] = None

    transform_functions: Optional[List[nn.Module]] = None

    def __post_init__(self) -> None:
        conds = (
            self.sort_by_pos,
            self.shuffle_before_sort_by_label,
            self.sort_by_pos_before_sort_by_label,
        )
        if not any(conds):
            raise ValueError(
                "At least one of sort_by_pos, shuffle_before_sort_by_label, "
                "or sort_by_pos_before_sort_by_label must be True."
            )

        self.transform_functions = self._config_base_transform()

    @property
    def transform(self) -> T.Compose:
        return T.Compose(self.transform_functions)

    def _config_base_transform(self) -> List[nn.Module]:
        transform_functions: List[nn.Module] = []
        if self.sort_by_pos:
            transform_functions.append(LexicographicSort())
        else:
            if self.shuffle_before_sort_by_label:
                transform_functions.append(ShuffleElements())
            elif self.sort_by_pos_before_sort_by_label:
                transform_functions.append(LexicographicSort())
            transform_functions.append(LabelDictSort(self.dataset_config.id2label))
        transform_functions.append(
            DiscretizeBoundingBox(
                num_x_grid=self.dataset_config.canvas_width,
                num_y_grid=self.dataset_config.canvas_height,
            )
        )
        return transform_functions

    def __call__(self, data: LayoutData) -> ProcessedLayoutData:
        assert self.transform is not None and self.return_keys is not None
        _data = self.transform(copy.deepcopy(data))
        return {k: _data[k] for k in self.return_keys}  # type: ignore
