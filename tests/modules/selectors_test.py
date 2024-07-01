import copy
from typing import List, Type

import pandas as pd
import pytest

from layout_prompter.dataset_configs import (
    LayoutDatasetConfig,
    PosterLayoutDatasetConfig,
    PubLayNetDatasetConfig,
    RicoDatasetConfig,
    WebUIDatasetConfig,
)
from layout_prompter.modules import create_selector
from layout_prompter.modules.selectors import (
    SELECTOR_MAP,
    CompletionExemplarSelector,
    ContentAwareExemplarSelector,
    ExemplarSelector,
    GenRelationExemplarSelector,
    GenTypeExemplarSelector,
    GenTypeSizeExemplarSelector,
    RefinementExemplarSelector,
    TextToLayoutExemplarSelector,
)
from layout_prompter.preprocessors import create_processor
from layout_prompter.testing import LayoutPrompterTestCase
from layout_prompter.typehint import Task


class TestCreateSelector(LayoutPrompterTestCase):
    @pytest.fixture
    def num_prompt(self) -> int:
        return 10

    @pytest.fixture
    def candidate_size(self) -> int:
        return -1

    @pytest.fixture
    def filenum_list(self) -> List[int]:
        return list(range(0, 5001, 1000))

    def test_selector_map(self):
        assert len(SELECTOR_MAP) == 7

    @pytest.mark.parametrize(
        argnames="dataset_config",
        argvalues=(RicoDatasetConfig(), PubLayNetDatasetConfig()),
    )
    @pytest.mark.parametrize(
        argnames="task, expected_selector_type",
        argvalues=(
            ("gen-t", GenTypeExemplarSelector),
            ("gen-ts", GenTypeSizeExemplarSelector),
            ("gen-r", GenRelationExemplarSelector),
            ("completion", CompletionExemplarSelector),
            ("refinement", RefinementExemplarSelector),
        ),
    )
    def test_constraint_explicit(
        self,
        task: Task,
        dataset_config: LayoutDatasetConfig,
        candidate_size: int,
        num_prompt: int,
        expected_selector_type: Type[ExemplarSelector],
        filenum_list: List[int],
    ):
        processor = create_processor(
            dataset_config=dataset_config,
            task=task,
        )
        raw_dataset = [
            self.load_raw_data(dataset_name=dataset_config.name, filenum=filenum)
            for filenum in filenum_list
        ]
        processed_dataset = [
            processor(copy.deepcopy(raw_data)) for raw_data in raw_dataset
        ]
        selector = create_selector(
            task=task,
            train_dataset=processed_dataset,
            candidate_size=candidate_size,
            num_prompt=num_prompt,
        )
        assert isinstance(selector, expected_selector_type)

    @pytest.mark.parametrize(
        argnames="task, expected_selector_type, dataset_config",
        argvalues=(
            ("content", ContentAwareExemplarSelector, PosterLayoutDatasetConfig()),
        ),
    )
    def test_content_aware(
        self,
        task: Task,
        dataset_config: LayoutDatasetConfig,
        candidate_size: int,
        num_prompt: int,
        expected_selector_type: Type[ExemplarSelector],
        filenum_list: List[int],
    ):
        metadata_path = self.FIXTURES_ROOT / dataset_config.name / "metadata_small.csv"
        metadata = pd.read_csv(metadata_path)

        processor = create_processor(
            dataset_config=dataset_config,
            task=task,
            metadata=metadata,
        )
        saliency_map_paths = [
            self.FIXTURES_ROOT
            / dataset_config.name
            / "raw"
            / f"{filenum}_mask_pred.png"
            for filenum in filenum_list
        ]
        processed_dataset = [
            processor(saliency_map_path=saliency_map_path, idx=filenum, split="train")
            for saliency_map_path, filenum in zip(saliency_map_paths, filenum_list)
        ]
        create_selector_kwargs = {
            "task": task,
            "train_dataset": processed_dataset,
            "candidate_size": candidate_size,
            "num_prompt": num_prompt,
        }
        with pytest.raises(AssertionError):
            create_selector(**create_selector_kwargs)

        selector = create_selector(
            **create_selector_kwargs, dataset_config=dataset_config
        )
        assert isinstance(selector, expected_selector_type)

    @pytest.mark.parametrize(
        argnames="task, expected_selector_type, dataset_config",
        argvalues=(("text", TextToLayoutExemplarSelector, WebUIDatasetConfig()),),
    )
    def test_text_to_layout(
        self,
        task: Task,
        dataset_config: LayoutDatasetConfig,
        candidate_size: int,
        num_prompt: int,
        expected_selector_type: Type[ExemplarSelector],
        filenum_list: List[int],
    ):
        processor = create_processor(
            dataset_config=dataset_config,
            task=task,
        )
        raw_dataset = [
            self.load_raw_data(dataset_name=dataset_config.name, filenum=filenum)
            for filenum in filenum_list
        ]
        processed_dataset = [
            processor(copy.deepcopy(raw_data)) for raw_data in raw_dataset
        ]
        selector = create_selector(
            task=task,
            train_dataset=processed_dataset,
            candidate_size=candidate_size,
            num_prompt=num_prompt,
        )
        assert isinstance(selector, expected_selector_type)
