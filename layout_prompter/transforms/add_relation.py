from __future__ import annotations

import logging
import random
from itertools import combinations, product
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from layout_prompter.utils import detect_loc_relation, detect_size_relation

logger = logging.getLogger(__name__)


class RelationTypes(object):
    types: List[str] = [
        "smaller",
        "equal",
        "larger",
        "top",
        "center",
        "bottom",
        "left",
        "right",
    ]
    _type2index: Optional[Dict[str, int]] = None
    _index2type: Optional[Dict[int, str]] = None

    @classmethod
    def type2index(cls) -> Dict[str, int]:
        if cls._type2index is None:
            cls._type2index = dict()
            for idx, cls_type in enumerate(cls.types):
                cls._type2index[cls_type] = idx
        return cls._type2index

    @classmethod
    def index2type(cls) -> Dict[int, str]:
        if cls._index2type is None:
            cls._index2type = dict()
            for idx, cls_type in enumerate(cls.types):
                cls._index2type[idx] = cls_type
        return cls._index2type


class AddRelation(nn.Module):
    def __init__(self, ratio: float = 0.1) -> None:
        super().__init__()

        self.ratio = ratio
        self.type2index = RelationTypes.type2index()

    def __call__(self, data):
        logger.debug(f"Before AddRelation:\n{data}")
        data["labels_with_canvas_index"] = [0] + list(
            range(len(data["labels_with_canvas"]) - 1)
        )
        N = len(data["labels_with_canvas"])

        rel_all = list(product(range(2), combinations(range(N), 2)))
        # size = min(int(len(rel_all)                     * self.ratio), 10)
        size = int(len(rel_all) * self.ratio)
        rel_sample = set(random.sample(rel_all, size))

        relations = []
        for i, j in combinations(range(N), 2):
            bi, bj = data["bboxes_with_canvas"][i], data["bboxes_with_canvas"][j]
            canvas = data["labels_with_canvas"][i] == 0

            if ((0, (i, j)) in rel_sample) and (not canvas):
                rel_size = detect_size_relation(bi, bj)
                relations.append(
                    [
                        data["labels_with_canvas"][i],
                        data["labels_with_canvas_index"][i],
                        data["labels_with_canvas"][j],
                        data["labels_with_canvas_index"][j],
                        self.type2index[rel_size],
                    ]
                )

            if (1, (i, j)) in rel_sample:
                rel_loc = detect_loc_relation(bi, bj, canvas)
                relations.append(
                    [
                        data["labels_with_canvas"][i],
                        data["labels_with_canvas_index"][i],
                        data["labels_with_canvas"][j],
                        data["labels_with_canvas_index"][j],
                        self.type2index[rel_loc],
                    ]
                )

        data["relations"] = torch.as_tensor(relations).long()
        logger.debug(f"After AddRelation:\n{data}")
        return data
