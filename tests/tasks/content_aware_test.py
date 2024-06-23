import logging
import os
from typing import Dict, List

import evaluate
import pandas as pd
import pytest
import torch
from tqdm.auto import tqdm

from layout_prompter import LayoutPrompter
from layout_prompter.dataset_configs import PosterLayoutDatasetConfig
from layout_prompter.modules import (
    ExemplarSelector,
    GPTCallar,
    Ranker,
    Serializer,
    create_selector,
    create_serializer,
)
from layout_prompter.parsers import GPTResponseParser
from layout_prompter.preprocessors import create_processor
from layout_prompter.testing import LayoutPrompterTestCase
from layout_prompter.typehint import InOutFormat, Task
from layout_prompter.utils import get_raw_data_path, read_pt, write_pt
from layout_prompter.visualizers import ContentAwareVisualizer, create_image_grid

logger = logging.getLogger(__name__)


def get_processed_dataset(
    processor,
    split: str,
    dataset_config: PosterLayoutDatasetConfig,
    base_dir: os.PathLike,
):
    filename = os.path.join(
        base_dir, "dataset", dataset_config.name, f"{split}_processed_data.pkl"
    )
    if os.path.exists(filename):
        processed_data = read_pt(filename)
    else:
        processed_data = []
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        raw_path = os.path.join(
            get_raw_data_path(dataset_config), split, "saliencymaps_pfpn"
        )
        raw_data = os.listdir(raw_path)
        raw_data = sorted(raw_data, key=lambda x: int(x.split("_")[0]))
        for rd in tqdm(raw_data, desc=f"{split} data processing..."):
            idx = int(rd.split("_")[0])
            data = processor(os.path.join(raw_path, rd), idx, split)
            if data:
                processed_data.append(data)
        write_pt(filename, processed_data)
    return processed_data


def get_padded_bbox_and_mask(
    bbox: torch.Tensor, max_elements: int
) -> Dict[str, torch.Tensor]:
    num_elements, num_attributes = bbox.shape
    padded_bbox = torch.zeros((max_elements, num_attributes))
    padded_bbox[:num_elements, :num_attributes] = bbox

    padded_mask = torch.zeros((max_elements,), dtype=torch.bool)
    padded_mask[:num_elements] = True

    return {"bbox": padded_bbox, "mask": padded_mask}


def get_batch_padded_bbox_and_mask(
    winner_layouts: List[Dict[str, torch.Tensor]], max_elements: int
) -> Dict[str, List[torch.Tensor]]:
    batch_dict: Dict[str, List[torch.Tensor]] = {"bbox": [], "mask": []}
    for winner_layout in winner_layouts:
        bbox = winner_layout["bboxes"]
        padded_dict = get_padded_bbox_and_mask(bbox=bbox, max_elements=max_elements)
        for k in padded_dict.keys():
            batch_dict[k].append(padded_dict[k])

    return batch_dict


class ProcessorFixtureClass(LayoutPrompterTestCase):
    @pytest.fixture(scope="module")
    def task(self) -> str:
        return "content"

    @pytest.fixture(scope="module")
    def dataset_config(self) -> PosterLayoutDatasetConfig:
        return PosterLayoutDatasetConfig()

    @pytest.fixture(scope="module")
    def metadata(self, dataset_config: PosterLayoutDatasetConfig):
        base_dir = get_raw_data_path(dataset_config)
        metadata_filepath = os.path.join(base_dir, "train_csv_9973.csv")
        return pd.read_csv(metadata_filepath)

    @pytest.fixture(scope="module")
    def processor(
        self,
        dataset_config: PosterLayoutDatasetConfig,
        task: Task,
        metadata: pd.DataFrame,
    ):
        return create_processor(
            dataset_config=dataset_config, task=task, metadata=metadata
        )


class ProcessedDatasetFixtureClass(ProcessorFixtureClass):
    @pytest.fixture(scope="class")
    def processed_dataset(self, processor, dataset_config):
        return {
            split: get_processed_dataset(
                processor=processor,
                split=split,
                dataset_config=dataset_config,
                base_dir=self.PROJECT_ROOT,
            )
            for split in ("train", "test")
        }

    @pytest.fixture(scope="class")
    def test_index(self):
        return 0

    @pytest.fixture(scope="class")
    def test_data(self, processed_dataset, test_index: int):
        processed_test_dataset = processed_dataset["test"]
        return processed_test_dataset[test_index]


class SelectorFixtureClass(ProcessedDatasetFixtureClass):
    @pytest.fixture(scope="class")
    def candidate_size(self) -> int:
        return -1  # -1 represents the complete training set

    @pytest.fixture(scope="class")
    def num_prompt(self) -> int:
        return 10

    @pytest.fixture(scope="class")
    def selector(
        self,
        task: Task,
        processed_dataset,
        candidate_size: int,
        num_prompt: int,
        dataset_config: PosterLayoutDatasetConfig,
    ):
        processed_train_dataset = processed_dataset["train"]
        return create_selector(
            task=task,
            train_dataset=processed_train_dataset,
            candidate_size=candidate_size,
            num_prompt=num_prompt,
            dataset_config=dataset_config,
        )


class SerializerFixtureClass(SelectorFixtureClass):
    @pytest.fixture(scope="class")
    def input_format(self) -> str:
        return "seq"

    @pytest.fixture(scope="class")
    def output_format(self) -> str:
        return "html"

    @pytest.fixture(scope="class")
    def add_index_token(self) -> bool:
        return True

    @pytest.fixture(scope="class")
    def add_unk_token(self) -> bool:
        return False

    @pytest.fixture(scope="class")
    def add_sep_token(self) -> bool:
        return True

    @pytest.fixture(scope="class")
    def serializer(
        self,
        dataset_config: PosterLayoutDatasetConfig,
        task: Task,
        input_format: InOutFormat,
        output_format: InOutFormat,
        add_unk_token: bool,
        add_sep_token: bool,
        add_index_token: bool,
    ):
        return create_serializer(
            dataset_config=dataset_config,
            task=task,
            input_format=input_format,
            output_format=output_format,
            add_unk_token=add_unk_token,
            add_sep_token=add_sep_token,
            add_index_token=add_index_token,
        )


class ParserFixtureClass(SerializerFixtureClass):
    @pytest.fixture(scope="class")
    def parser(
        self, dataset_config: PosterLayoutDatasetConfig, output_format: InOutFormat
    ):
        return GPTResponseParser(
            dataset_config=dataset_config,
            output_format=output_format,
        )


class LLMFixtureClass(ParserFixtureClass):
    @pytest.fixture(scope="class")
    def model(self) -> str:
        return "gpt-4o"

    @pytest.fixture(scope="class")
    def max_tokens(self) -> int:
        return 800

    @pytest.fixture(scope="class")
    def temperature(self) -> float:
        return 0.7

    @pytest.fixture(scope="class")
    def top_p(self) -> float:
        return 1.0

    @pytest.fixture(scope="class")
    def frequency_penalty(self) -> float:
        return 0.0

    @pytest.fixture(scope="class")
    def presence_penalty(self) -> float:
        return 0.0

    @pytest.fixture(scope="class")
    def num_return(self) -> int:
        return 10

    @pytest.fixture(scope="class")
    def stop_token(self) -> str:
        return "\n\n"

    @pytest.fixture(scope="class")
    def llm(
        self,
        parser: GPTResponseParser,
        model: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        frequency_penalty: float,
        presence_penalty: float,
        num_return: int,
        stop_token: str,
    ):
        return GPTCallar(
            parser=parser,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            num_return=num_return,
            stop_token=stop_token,
        )


class RankerFixtureClass(LLMFixtureClass):
    @pytest.fixture(scope="class")
    def ranker(self) -> Ranker:
        return Ranker()


class TestContentAware(RankerFixtureClass):
    def test_pipeline(
        self,
        dataset_config: PosterLayoutDatasetConfig,
        selector: ExemplarSelector,
        serializer: Serializer,
        llm: GPTCallar,
        ranker: Ranker,
        test_data,
        test_index: int,
    ):
        pipeline = LayoutPrompter(
            serializer=serializer, selector=selector, llm=llm, ranker=ranker
        )

        exemplars = pipeline.get_exemplars(test_data=test_data)
        ranked_response = pipeline(test_data=test_data, exemplars=exemplars)

        canvas_path = os.path.join(
            get_raw_data_path(dataset_config=dataset_config), "test", "image_canvas"
        )
        visualizer = ContentAwareVisualizer(
            dataset_config=dataset_config, canvas_path=canvas_path
        )
        images = visualizer(ranked_response, test_idx=test_index)
        create_image_grid(images)

    @pytest.fixture
    def evaluation_metrics(self) -> List[str]:
        return [
            "creative-graphic-design/layout-alignment",
            "creative-graphic-design/layout-overlap",
        ]

    @pytest.fixture
    def max_elements(self) -> int:
        # https://github.com/PKU-ICST-MIPL/PosterLayout-CVPR2023/blob/main/main.py#L156
        return 32

    def test_pipeline_all(
        self,
        selector: ExemplarSelector,
        serializer: Serializer,
        llm: GPTCallar,
        ranker: Ranker,
        processed_dataset,
        evaluation_metrics: List[str],
        max_elements: int,
    ):
        pipeline = LayoutPrompter(
            serializer=serializer, selector=selector, llm=llm, ranker=ranker
        )

        generated_layouts_list = []
        for test_data in tqdm(processed_dataset["test"][:5]):
            try:
                exemplars = pipeline.get_exemplars(test_data=test_data)
                ranked_response = pipeline(test_data=test_data, exemplars=exemplars)
                generated_layouts_list.append(ranked_response)

            except Exception as err:
                logger.warning(err)
                generated_layouts_list.append(None)

        metrics = evaluate.combine(evaluations=evaluation_metrics)
        winner_layouts = [layouts[0] for layouts in generated_layouts_list]

        batch_dict = get_batch_padded_bbox_and_mask(
            winner_layouts=winner_layouts, max_elements=max_elements
        )
        batch_bbox = torch.stack(batch_dict["bbox"])
        batch_mask = torch.stack(batch_dict["mask"])

        eval_results = metrics.compute(bbox=batch_bbox, mask=batch_mask)
        eval_results = {
            k: {"mean": v.mean(), "std": v.std()} for k, v in eval_results.items()
        }
        logger.info(eval_results)
