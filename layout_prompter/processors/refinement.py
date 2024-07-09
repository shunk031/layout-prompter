from dataclasses import dataclass
from typing import Tuple

from layout_prompter.processors import TaskProcessor
from layout_prompter.transforms import AddGaussianNoise


@dataclass
class RefinementProcessor(TaskProcessor):
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

    gaussian_noise_mean: float = 0.0
    gaussian_noise_std: float = 0.01
    train_bernoulli_beta: float = 1.0

    def __post_init__(self) -> None:
        super().__post_init__()
        assert self.transform_functions is not None

        self.transform_functions = [
            AddGaussianNoise(
                mean=self.gaussian_noise_mean,
                std=self.gaussian_noise_std,
                bernoulli_beta=self.train_bernoulli_beta,
            )
        ] + self.transform_functions
