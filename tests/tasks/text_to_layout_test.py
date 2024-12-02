import json
import logging
import os
from typing import get_args

import openai_responses
import pytest
from openai import OpenAI
from openai_responses import OpenAIMock
from PIL import ImageChops
from tqdm.auto import tqdm

from layout_prompter.modules import (
    Ranker,
    build_prompt,
    create_selector,
    create_serializer,
)
from layout_prompter.parsers import Parser
from layout_prompter.preprocessors import create_processor
from layout_prompter.testing import LayoutPrompterTestCase
from layout_prompter.typehint import TextToLayoutDataset
from layout_prompter.utils import RAW_DATA_PATH, read_json, read_pt, write_pt
from layout_prompter.visualizers import Visualizer, create_image_grid

logger = logging.getLogger(__name__)


class TestTextToLayout(LayoutPrompterTestCase):
    @pytest.fixture
    def task(self) -> str:
        return "text"

    @pytest.fixture
    def add_unk_token(self) -> bool:
        return False

    @pytest.fixture
    def add_index_token(self) -> bool:
        return False

    @pytest.fixture
    def add_sep_token(self) -> bool:
        return True

    @pytest.fixture
    def max_tokens(self) -> int:
        return 1200

    @openai_responses.mock()
    @pytest.mark.parametrize(
        argnames="dataset",
        argvalues=get_args(TextToLayoutDataset),
    )
    @pytest.mark.parametrize(
        argnames="test_idx",
        argvalues=list(range(5)),
    )
    def test_text_to_layout(
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
    ) -> None:
        processor = create_processor(dataset=dataset, task=task)
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
                raw_path = os.path.join(RAW_DATA_PATH(dataset), f"{split}.json")
                raw_data = read_json(raw_path)
                for rd in tqdm(raw_data, desc=f"{split} data processing..."):
                    processed_data.append(processor(rd))
                write_pt(filename, processed_data)
            return processed_data

        processed_train_data = get_processed_data("train")
        # processed_val_data = get_processed_data("val")
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

        # temperature = 0.7
        # max_tokens = 1200
        # top_p = 1
        # frequency_penalty = 0
        # presence_penalty = 0
        # num_return = 10
        # stop_token = "\n\n"

        mock_json_dir = self.FIXTURES_ROOT / "text_to_layout" / dataset
        mock_json_dir.mkdir(parents=True, exist_ok=True)

        mock_json_path = (
            self.FIXTURES_ROOT / "text_to_layout" / dataset / f"{test_idx=}.json"
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

        parser = Parser(dataset=dataset, output_format=output_format)
        parsed_response = parser(response)
        print(f"filter {num_return - len(parsed_response)} invalid response")

        ranker = Ranker()
        ranked_response = ranker(parsed_response)

        visualizer = Visualizer(dataset)
        images = visualizer(ranked_response)
        image = create_image_grid(images)

        expected_image_path = (
            self.FIXTURES_ROOT / "text_to_layout" / dataset / f"{test_idx=}.png"
        )
        expected_image = self._load_image(expected_image_path)

        diff = ImageChops.difference(image, expected_image)
        assert diff.getbbox() is None

        # image.save(expected_image_path)
