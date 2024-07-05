from __future__ import annotations

import abc
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Type

import cv2
import numpy as np

from layout_prompter.configs import LayoutDatasetConfig
from layout_prompter.typehint import Task
from layout_prompter.utils import (
    labels_bboxes_similarity,
    labels_similarity,
)

if TYPE_CHECKING:
    from layout_prompter.typehint import ProcessedLayoutData, Task

__all__ = [
    "ExemplarSelector",
    "GenTypeExemplarSelector",
    "GenTypeSizeExemplarSelector",
    "GenRelationExemplarSelector",
    "CompletionExemplarSelector",
    "RefinementExemplarSelector",
    "ContentAwareExemplarSelector",
    "TextToLayoutExemplarSelector",
    "create_selector",
]


@dataclass
class ExemplarSelector(object, metaclass=abc.ABCMeta):
    train_dataset: List[ProcessedLayoutData] = field(repr=False)
    candidate_size: int
    num_prompt: int
    shuffle: bool = True
    dataset_config: Optional[LayoutDatasetConfig] = None

    def __post_init__(self) -> None:
        if self.candidate_size > 0:
            random.shuffle(self.train_dataset)
            self.train_dataset = self.train_dataset[: self.candidate_size]

    @abc.abstractmethod
    def __call__(self, test_data: ProcessedLayoutData) -> List[ProcessedLayoutData]:
        raise NotImplementedError

    def _is_filter(self, data: ProcessedLayoutData) -> bool:
        return (data["discrete_gold_bboxes"][:, 2:] == 0).sum().bool().item()  # type: ignore

    def _retrieve_exemplars(self, scores: list) -> List[ProcessedLayoutData]:
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        exemplars = []
        for i in range(len(self.train_dataset)):
            if not self._is_filter(self.train_dataset[scores[i][0]]):
                exemplars.append(self.train_dataset[scores[i][0]])
                if len(exemplars) == self.num_prompt:
                    break
        if self.shuffle:
            random.shuffle(exemplars)
        return exemplars


@dataclass
class GenTypeExemplarSelector(ExemplarSelector):
    def __call__(self, test_data: ProcessedLayoutData) -> List[ProcessedLayoutData]:
        scores = []
        test_labels = test_data["labels"]
        for i in range(len(self.train_dataset)):
            train_labels = self.train_dataset[i]["labels"]
            score = labels_similarity(train_labels, test_labels)
            scores.append([i, score])
        return self._retrieve_exemplars(scores)


@dataclass
class GenTypeSizeExemplarSelector(ExemplarSelector):
    labels_weight: float = 0.5
    bboxes_weight: float = 0.5

    def __call__(self, test_data: ProcessedLayoutData) -> List[ProcessedLayoutData]:
        scores = []
        test_labels = test_data["labels"]
        test_bboxes = test_data["bboxes"][:, 2:]
        for i in range(len(self.train_dataset)):
            train_labels = self.train_dataset[i]["labels"]
            train_bboxes = self.train_dataset[i]["bboxes"][:, 2:]
            score = labels_bboxes_similarity(
                train_labels,
                train_bboxes,
                test_labels,
                test_bboxes,
                self.labels_weight,
                self.bboxes_weight,
            )
            scores.append([i, score])
        return self._retrieve_exemplars(scores)


@dataclass
class GenRelationExemplarSelector(ExemplarSelector):
    def __call__(self, test_data: ProcessedLayoutData) -> List[ProcessedLayoutData]:
        scores = []
        test_labels = test_data["labels"]
        for i in range(len(self.train_dataset)):
            train_labels = self.train_dataset[i]["labels"]
            score = labels_similarity(train_labels, test_labels)
            scores.append([i, score])
        return self._retrieve_exemplars(scores)


@dataclass
class CompletionExemplarSelector(ExemplarSelector):
    labels_weight: float = 0.0
    bboxes_weight: float = 1.0

    def __call__(self, test_data: ProcessedLayoutData) -> List[ProcessedLayoutData]:
        scores = []
        test_labels = test_data["labels"][:1]
        test_bboxes = test_data["bboxes"][:1, :]
        for i in range(len(self.train_dataset)):
            train_labels = self.train_dataset[i]["labels"][:1]
            train_bboxes = self.train_dataset[i]["bboxes"][:1, :]
            score = labels_bboxes_similarity(
                bboxes_1=train_bboxes,
                bboxes_2=test_bboxes,
                bboxes_weight=self.bboxes_weight,
                labels_1=train_labels,
                labels_2=test_labels,
                labels_weight=self.labels_weight,
            )
            scores.append([i, score])
        return self._retrieve_exemplars(scores)


@dataclass
class RefinementExemplarSelector(ExemplarSelector):
    labels_weight: float = 0.5
    bboxes_weight: float = 0.5

    def __call__(self, test_data: ProcessedLayoutData) -> List[ProcessedLayoutData]:
        scores = []
        test_labels = test_data["labels"]
        test_bboxes = test_data["bboxes"]
        for i in range(len(self.train_dataset)):
            train_labels = self.train_dataset[i]["labels"]
            train_bboxes = self.train_dataset[i]["bboxes"]
            score = labels_bboxes_similarity(
                train_labels,
                train_bboxes,
                test_labels,
                test_bboxes,
                self.labels_weight,
                self.bboxes_weight,
            )
            scores.append([i, score])
        return self._retrieve_exemplars(scores)


@dataclass
class ContentAwareExemplarSelector(ExemplarSelector):
    def _to_binary_image(self, content_bboxes):
        assert self.dataset_config is not None

        binary_image = np.zeros(
            (self.dataset_config.canvas_height, self.dataset_config.canvas_width),
            dtype=np.uint8,
        )
        content_bboxes = content_bboxes.tolist()
        for content_bbox in content_bboxes:
            left, top, width, height = content_bbox
            cv2.rectangle(
                binary_image,
                (left, top),
                (left + width, top + height),
                255,
                thickness=-1,
            )
        return binary_image

    def __call__(self, test_data: ProcessedLayoutData) -> List[ProcessedLayoutData]:
        scores = []
        test_content_bboxes = test_data["discrete_content_bboxes"]
        test_binary = self._to_binary_image(test_content_bboxes)
        for i in range(len(self.train_dataset)):
            train_content_bboxes = self.train_dataset[i]["discrete_content_bboxes"]
            train_binary = self._to_binary_image(train_content_bboxes)
            intersection = cv2.bitwise_and(train_binary, test_binary)
            union = cv2.bitwise_or(train_binary, test_binary)
            iou = (np.sum(intersection) + 1) / (np.sum(union) + 1)
            scores.append([i, iou])
        return self._retrieve_exemplars(scores)


@dataclass
class TextToLayoutExemplarSelector(ExemplarSelector):
    def __call__(self, test_data: ProcessedLayoutData) -> List[ProcessedLayoutData]:
        scores = []
        test_embedding = test_data["embedding"]
        for i in range(len(self.train_dataset)):
            train_embedding = self.train_dataset[i]["embedding"]
            score = (train_embedding @ test_embedding.T).item()
            scores.append([i, score])
        return self._retrieve_exemplars(scores)


SELECTOR_MAP: Dict[Task, Type[ExemplarSelector]] = {
    "gen-t": GenTypeExemplarSelector,
    "gen-ts": GenTypeSizeExemplarSelector,
    "gen-r": GenRelationExemplarSelector,
    "completion": CompletionExemplarSelector,
    "refinement": RefinementExemplarSelector,
    "content": ContentAwareExemplarSelector,
    "text": TextToLayoutExemplarSelector,
}


def create_selector(
    task: Task,
    train_dataset: List[ProcessedLayoutData],
    candidate_size: int,
    num_prompt: int,
    dataset_config: Optional[LayoutDatasetConfig] = None,
) -> ExemplarSelector:
    if task == "content":
        assert (
            dataset_config is not None
        ), "`dataset` must be provided for content-aware task"

    selector_cls = SELECTOR_MAP[task]

    selector = selector_cls(
        train_dataset=train_dataset,
        candidate_size=candidate_size,
        num_prompt=num_prompt,
        dataset_config=dataset_config,
    )
    return selector
