import json
import logging
import os
from typing import get_args

import openai_responses
import pandas as pd
import pytest
from openai import OpenAI
from openai_responses import OpenAIMock
from PIL import ImageChops
from tqdm import tqdm

from layout_prompter.parsing import Parser
from layout_prompter.preprocess import create_processor
from layout_prompter.ranker import Ranker
from layout_prompter.selection import create_selector
from layout_prompter.serialization import build_prompt, create_serializer
from layout_prompter.testing import LayoutPrompterTestCase
from layout_prompter.typehint import ContentAwareDataset
from layout_prompter.utils import RAW_DATA_PATH, read_pt, write_pt
from layout_prompter.visualization import ContentAwareVisualizer, create_image_grid

logger = logging.getLogger(__name__)


class TestContentAwareCase(LayoutPrompterTestCase):
    @pytest.fixture
    def task(self) -> str:
        return "content"

    @pytest.fixture
    def metadata(self, request) -> pd.DataFrame:
        dataset = request.param
        return pd.read_csv(os.path.join(RAW_DATA_PATH(dataset), "train_csv_9973.csv"))

    @pytest.fixture
    def add_unk_token(self) -> bool:
        return False

    @pytest.fixture
    def add_index_token(self) -> bool:
        return True

    @pytest.fixture
    def add_sep_token(self) -> bool:
        return True

    @pytest.fixture
    def processor(self, dataset: str, task: str, metadata):
        return create_processor(dataset, task, metadata=metadata)

    @openai_responses.mock()
    @pytest.mark.parametrize(
        argnames="dataset",
        argvalues=get_args(ContentAwareDataset),
    )
    @pytest.mark.parametrize(
        argnames="metadata",
        argvalues=get_args(ContentAwareDataset),
        indirect=True,
    )
    @pytest.mark.parametrize(
        argnames="test_idx",
        argvalues=list(range(5)),
    )
    def test_content_aware(
        self,
        #
        # Mock configurations
        #
        openai_mock: OpenAIMock,
        #
        # Test configurations
        #
        dataset: str,
        task: str,
        input_format: str,
        output_format: str,
        add_unk_token: bool,
        add_index_token: bool,
        add_sep_token: bool,
        candidate_size: int,
        num_prompt: int,
        test_idx: int,
        #
        # Model configurations
        #
        model: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        frequency_penalty: float,
        presence_penalty: float,
        num_return: int,
        stop_token: str,
        #
        # Module configuration
        #
        processor,
    ) -> None:
        # metadata = read_csv(os.path.join(RAW_DATA_PATH(dataset), "train_csv_9973.csv"))
        # processor = create_processor(dataset, task, metadata=metadata)
        base_dir = os.path.dirname(os.getcwd())

        def get_processed_data(split):
            filename = os.path.join(
                base_dir, "dataset", dataset, "processed", task, f"{split}.pt"
            )
            if os.path.exists(filename):
                processed_data = read_pt(filename)
            else:
                processed_data = []
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                raw_path = os.path.join(
                    RAW_DATA_PATH(dataset), split, "saliencymaps_pfpn"
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

        processed_train_data = get_processed_data("train")
        processed_test_data = get_processed_data("test")

        selector = create_selector(
            task=task,
            train_data=processed_train_data,
            candidate_size=candidate_size,
            num_prompt=num_prompt,
        )

        exemplars = selector(processed_test_data[test_idx])

        serializer = create_serializer(
            dataset=dataset,
            task=task,
            input_format=input_format,
            output_format=output_format,
            add_index_token=add_index_token,
            add_sep_token=add_sep_token,
            add_unk_token=add_unk_token,
        )
        prompt = build_prompt(
            serializer, exemplars, processed_test_data[test_idx], dataset
        )

        mock_json_path = (
            self.FIXTURES_ROOT / "content_aware" / dataset / f"{test_idx=}.json"
        )
        with mock_json_path.open("r") as rf:
            openai_mock.chat.completions.create.response = json.load(rf)

        client = OpenAI()

        messages = [
            {"role": "system", "content": prompt["system_prompt"]},
            {"role": "user", "content": prompt["user_prompt"]},
        ]
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            n=num_return,
            stop=[stop_token],
        )

        # hoge = {"choices": []}
        # for choice in response.choices:
        #     hoge["choices"].append(
        #         {
        #             "index": choice.index,
        #             "finish_reason": choice.finish_reason,
        #             "message": {
        #                 "content": choice.message.content,
        #                 "role": choice.message.role,
        #             },
        #         }
        #     )
        # with open(f"{test_idx=}.json", "w") as wf:
        #     json.dump(hoge, wf, indent=4)

        parser = Parser(dataset=dataset, output_format=output_format)
        parsed_response = parser(response)
        logger.info(f"filter {num_return - len(parsed_response)} invalid response")

        ranker = Ranker()
        ranked_response = ranker(parsed_response)

        visualizer = ContentAwareVisualizer()
        images = visualizer(ranked_response, processed_test_data[test_idx]["idx"])
        image = create_image_grid(images)

        expected_image_path = (
            self.FIXTURES_ROOT / "content_aware" / dataset / f"{test_idx=}.png"
        )
        expected_image = self._load_image(expected_image_path)

        diff = ImageChops.difference(image, expected_image)
        assert diff.getbbox() is None
