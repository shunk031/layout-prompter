import json
import pathlib

import torch


class LayoutPrompterTestCase(object):
    PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
    MODULE_ROOT = PROJECT_ROOT / "layout_prompter"
    TEST_ROOT = PROJECT_ROOT / "tests"
    FIXTURES_ROOT = PROJECT_ROOT / "test_fixtures"

    def _convert_raw_to_tensor_dict(self, data):
        return {
            **data,
            "bboxes": torch.Tensor(data["bboxes"]),
            "labels": torch.Tensor(data["labels"]),
        }

    def _load_json(self, filepath: pathlib.Path):
        with filepath.open("r") as rf:
            json_dict = json.load(rf)
        return json_dict

    def load_raw_data(self, dataset_name: str, filenum: int):
        raw_data_file = self.FIXTURES_ROOT / dataset_name / "raw" / f"{filenum}.json"
        raw_data = self._load_json(raw_data_file)
        return self._convert_raw_to_tensor_dict(data=raw_data)

    def load_processed_data(self, dataset_name: str, task: str, filenum: int):
        processed_data_file = (
            self.FIXTURES_ROOT / dataset_name / task / f"{filenum}.json"
        )
        return self._load_json(processed_data_file)
