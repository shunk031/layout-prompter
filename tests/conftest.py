import logging
import random

import numpy as np
import pytest
import torch

logger = logging.getLogger(__name__)


@pytest.fixture
def seed() -> int:
    return 19950815


@pytest.fixture(autouse=True)
def set_seed(seed: int) -> None:
    logger.debug(f"Setting seed to {seed}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@pytest.fixture
def model() -> str:
    return "gpt-4o"


@pytest.fixture
def temperature() -> float:
    return 0.7


@pytest.fixture
def max_tokens() -> int:
    return 800


@pytest.fixture
def top_p() -> float:
    return 1.0


@pytest.fixture
def frequency_penalty() -> float:
    return 0.0


@pytest.fixture
def presence_penalty() -> float:
    return 0.0


@pytest.fixture
def num_return() -> int:
    return 10


@pytest.fixture
def stop_token() -> str:
    return "\n\n"


@pytest.fixture
def input_format() -> str:
    return "seq"


@pytest.fixture
def output_format() -> str:
    return "html"


@pytest.fixture
def candidate_size() -> int:
    # -1 represents the complete training set
    return -1


@pytest.fixture
def num_prompt() -> int:
    return 10
