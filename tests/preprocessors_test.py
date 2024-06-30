import copy
from typing import Type

import numpy as np
import pandas as pd
import pytest
import torch

from layout_prompter.dataset_configs import (
    LayoutDatasetConfig,
    PosterLayoutDatasetConfig,
    PubLayNetDatasetConfig,
    RicoDatasetConfig,
    WebUIDatasetConfig,
)
from layout_prompter.preprocessors import (
    PROCESSOR_MAP,
    CompletionProcessor,
    ContentAwareProcessor,
    GenRelationProcessor,
    GenTypeProcessor,
    GenTypeSizeProcessor,
    ProcessorMixin,
    RefinementProcessor,
    TextToLayoutProcessor,
    create_processor,
)
from layout_prompter.testing import LayoutPrompterTestCase
from layout_prompter.typehint import Task


class TestGenTypeProcessor(LayoutPrompterTestCase):
    TASK: Task = "gen-t"

    @pytest.mark.parametrize(
        argnames="dataset_config",
        argvalues=(RicoDatasetConfig(), PubLayNetDatasetConfig()),
    )
    @pytest.mark.parametrize(
        argnames="filenum",
        argvalues=(0, 1000, 2000, 3000, 4000, 5000),
    )
    def test_processor(self, dataset_config: LayoutDatasetConfig, filenum: int):
        raw_data = self.load_raw_data(
            dataset_name=dataset_config.name,
            filenum=filenum,
        )

        processor = create_processor(
            dataset_config=dataset_config,
            task=self.TASK,
        )
        processed_data = processor(copy.deepcopy(raw_data))

        expected_processed_data = self.load_processed_data(
            dataset_name=dataset_config.name,
            task=self.TASK,
            filenum=filenum,
        )
        assert set(processed_data.keys()) == set(expected_processed_data.keys())

        for k in processed_data.keys():
            assert processed_data[k].numpy().tolist() == expected_processed_data[k]  # type: ignore


class TestGenTypeSizeProcessor(LayoutPrompterTestCase):
    TASK: Task = "gen-ts"

    @pytest.mark.parametrize(
        argnames="filenum",
        argvalues=(0, 1000, 2000, 3000, 4000, 5000),
    )
    @pytest.mark.parametrize(
        argnames="dataset_config",
        argvalues=(RicoDatasetConfig(), PubLayNetDatasetConfig()),
    )
    def test_processor(self, dataset_config: LayoutDatasetConfig, filenum: int):
        raw_data = self.load_raw_data(
            dataset_name=dataset_config.name,
            filenum=filenum,
        )
        processor = create_processor(
            dataset_config=dataset_config,
            task=self.TASK,
        )
        processed_data = processor(copy.deepcopy(raw_data))

        expected_processed_data = self.load_processed_data(
            dataset_name=dataset_config.name,
            task=self.TASK,
            filenum=filenum,
        )
        assert set(processed_data.keys()) == set(expected_processed_data.keys())

        for k in processed_data.keys():
            assert processed_data[k].numpy().tolist() == expected_processed_data[k]  # type: ignore


class TestGenRelationProcessor(LayoutPrompterTestCase):
    TASK: Task = "gen-r"

    @pytest.mark.parametrize(
        argnames="dataset_config",
        argvalues=(RicoDatasetConfig(), PubLayNetDatasetConfig()),
    )
    @pytest.mark.parametrize(
        argnames="filenum",
        argvalues=(0, 1000, 2000, 3000, 4000, 5000),
    )
    def test_processor(self, dataset_config: LayoutDatasetConfig, filenum: int):
        print("Start test_processor")
        raw_data = self.load_raw_data(
            dataset_name=dataset_config.name,
            filenum=filenum,
        )

        processor = create_processor(
            dataset_config=dataset_config,
            task=self.TASK,
        )
        processed_data = processor(copy.deepcopy(raw_data))

        expected_processed_data = self.load_processed_data(
            dataset_name=dataset_config.name,
            task=self.TASK,
            filenum=filenum,
        )
        assert set(processed_data.keys()) == set(expected_processed_data.keys())

        for k in processed_data.keys():
            assert processed_data[k].numpy().tolist() == expected_processed_data[k]  # type: ignore


class TestCompletionProcessor(LayoutPrompterTestCase):
    TASK: Task = "completion"

    @pytest.mark.parametrize(
        argnames="dataset_config",
        argvalues=(RicoDatasetConfig(), PubLayNetDatasetConfig()),
    )
    @pytest.mark.parametrize(
        argnames="filenum",
        argvalues=(0, 1000, 2000, 3000, 4000, 5000),
    )
    def test_processor(self, dataset_config: LayoutDatasetConfig, filenum: int):
        raw_data = self.load_raw_data(
            dataset_name=dataset_config.name,
            filenum=filenum,
        )
        processor = create_processor(
            dataset_config=dataset_config,
            task=self.TASK,
        )
        processed_data = processor(raw_data)

        expected_processed_data = self.load_processed_data(
            dataset_name=dataset_config.name,
            task=self.TASK,
            filenum=filenum,
        )
        assert set(processed_data.keys()) == set(expected_processed_data.keys())

        for k in processed_data.keys():
            assert processed_data[k].numpy().tolist() == expected_processed_data[k]  # type: ignore


class TestRefinementProcessor(LayoutPrompterTestCase):
    TASK: Task = "refinement"

    @pytest.mark.parametrize(
        argnames="dataset_config",
        argvalues=(RicoDatasetConfig(), PubLayNetDatasetConfig()),
    )
    @pytest.mark.parametrize(
        argnames="filenum",
        argvalues=(0, 1000, 2000, 3000, 4000, 5000),
    )
    def test_processor(self, dataset_config: LayoutDatasetConfig, filenum: int):
        raw_data = self.load_raw_data(
            dataset_name=dataset_config.name,
            filenum=filenum,
        )
        processor = create_processor(
            dataset_config=dataset_config,
            task=self.TASK,
        )
        processed_data = processor(copy.deepcopy(raw_data))

        expected_processed_data = self.load_processed_data(
            dataset_name=dataset_config.name,
            task=self.TASK,
            filenum=filenum,
        )
        assert set(processed_data.keys()) == set(expected_processed_data.keys())

        for k in processed_data.keys():
            processed_data_k = processed_data[k].numpy().tolist()
            expected_processed_data_k = expected_processed_data[k]
            assert np.allclose(processed_data_k, expected_processed_data_k, atol=1e-5)


class TestContentAwareProcessor(LayoutPrompterTestCase):
    TASK: Task = "content"

    def load_metadata(self, dataset_name: str) -> pd.DataFrame:
        metadata_path = self.FIXTURES_ROOT / dataset_name / "metadata_small.csv"
        return pd.read_csv(metadata_path)

    @pytest.mark.parametrize(
        argnames="dataset_config",
        argvalues=(PosterLayoutDatasetConfig(),),
    )
    @pytest.mark.parametrize(
        argnames="filenum",
        argvalues=(0, 1000, 2000, 3000, 4000, 5000),
    )
    def test_processor(self, dataset_config: LayoutDatasetConfig, filenum: int):
        metadata = self.load_metadata(dataset_name=dataset_config.name)

        processor = create_processor(
            dataset_config=dataset_config,
            task=self.TASK,
            metadata=metadata,
        )
        raw_data_path = (
            self.FIXTURES_ROOT
            / dataset_config.name
            / "raw"
            / f"{filenum}_mask_pred.png"
        )
        processed_data = processor(
            str(raw_data_path),
            idx=filenum,
            split="train",
        )
        expected_processed_data = self.load_processed_data(
            dataset_name=dataset_config.name,
            task=self.TASK,
            filenum=filenum,
        )
        assert set(processed_data.keys()) == set(expected_processed_data.keys())

        for k in processed_data.keys():
            processed_data_k = processed_data[k]
            expected_processed_data_k = expected_processed_data[k]

            if isinstance(processed_data_k, torch.Tensor):
                processed_data_k = processed_data_k.numpy().tolist()

            assert processed_data_k == expected_processed_data_k  # type: ignore


class TestTextToLayoutProcessor(LayoutPrompterTestCase):
    TASK: Task = "text"

    def _convert_raw_to_tensor_dict(self, data):
        # There is no need to convert the data to tensor in this task
        return data

    @pytest.mark.parametrize(
        argnames="dataset_config",
        argvalues=(WebUIDatasetConfig(),),
    )
    @pytest.mark.parametrize(
        argnames="filenum",
        argvalues=(0, 1000, 2000, 3000, 4000, 5000),
    )
    def test_processor(self, dataset_config: LayoutDatasetConfig, filenum: int):
        raw_data = self.load_raw_data(
            dataset_name=dataset_config.name,
            filenum=filenum,
        )
        processor = create_processor(
            dataset_config=dataset_config,
            task=self.TASK,
        )
        processed_data = processor(copy.deepcopy(raw_data))

        expected_processed_data = self.load_processed_data(
            dataset_name=dataset_config.name,
            task=self.TASK,
            filenum=filenum,
        )
        assert set(processed_data.keys()) == set(expected_processed_data.keys())

        for k in processed_data.keys():
            processed_data_k = processed_data[k]
            expected_processed_data_k = expected_processed_data[k]

            if isinstance(processed_data_k, str):
                assert processed_data_k == expected_processed_data_k
            elif isinstance(processed_data_k, torch.Tensor):
                processed_data_k = processed_data_k.numpy().tolist()
                assert np.allclose(
                    processed_data_k, expected_processed_data_k, atol=1e-5
                )
            else:
                raise ValueError(f"Unexpected type (k: {k}): {type(processed_data_k)}")


class TestCreateProcessor(LayoutPrompterTestCase):
    def test_processor_map(self):
        assert len(PROCESSOR_MAP) == 7

    @pytest.mark.parametrize(
        argnames="dataset_config",
        argvalues=(
            RicoDatasetConfig(),
            PubLayNetDatasetConfig(),
        ),
    )
    @pytest.mark.parametrize(
        argnames="task, expected_processor_type",
        argvalues=(
            ("gen-t", GenTypeProcessor),
            ("gen-ts", GenTypeSizeProcessor),
            ("gen-r", GenRelationProcessor),
            ("completion", CompletionProcessor),
            ("refinement", RefinementProcessor),
        ),
    )
    def test_constraint_explicit(
        self,
        dataset_config: LayoutDatasetConfig,
        task: Task,
        expected_processor_type: Type[ProcessorMixin],
    ):
        processor = create_processor(dataset_config=dataset_config, task=task)
        assert isinstance(processor, expected_processor_type)

    @pytest.mark.parametrize(
        argnames="task, expected_processor_type, dataset_config",
        argvalues=(("content", ContentAwareProcessor, PosterLayoutDatasetConfig()),),
    )
    def test_content_aware(
        self,
        task: Task,
        expected_processor_type: Type[ProcessorMixin],
        dataset_config: LayoutDatasetConfig,
    ):
        processor = create_processor(dataset_config=dataset_config, task=task)
        assert isinstance(processor, expected_processor_type)

    @pytest.mark.parametrize(
        argnames="task, expected_processor_type, dataset_config",
        argvalues=(("text", TextToLayoutProcessor, WebUIDatasetConfig()),),
    )
    def test_text_to_layout(
        self,
        task: Task,
        expected_processor_type: Type[ProcessorMixin],
        dataset_config: LayoutDatasetConfig,
    ):
        processor = create_processor(dataset_config=dataset_config, task=task)
        assert isinstance(processor, expected_processor_type)

    @pytest.mark.parametrize(
        argnames="dataset_config",
        argvalues=(
            RicoDatasetConfig(),
            PosterLayoutDatasetConfig(),
            PosterLayoutDatasetConfig(),
            WebUIDatasetConfig(),
        ),
    )
    def test_not_exist_processor(self, dataset_config: LayoutDatasetConfig):
        with pytest.raises(KeyError):
            create_processor(dataset_config=dataset_config, task="not_exist_task")  # type: ignore
