import logging
import random

import numpy as np
import pytest
import torch

logger = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    logger.debug(f"Setting seed to {seed}")
