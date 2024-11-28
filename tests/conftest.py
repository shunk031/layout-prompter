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
