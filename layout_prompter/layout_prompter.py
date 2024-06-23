from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Final, List, Optional

from tenacity import after_log, retry, stop_after_attempt

from layout_prompter.exception import LayoutPrompterException
from layout_prompter.modules import (
    LLM,
    ExemplarSelector,
    Ranker,
    RankerOutput,
    Serializer,
)

if TYPE_CHECKING:
    from layout_prompter.typehint import ProcessedLayoutData

logger = logging.getLogger(__name__)

DEFAULT_NUM_MAX_TRY: Final[int] = 5


@dataclass
class LayoutPrompter(object):
    serializer: Serializer
    selector: ExemplarSelector
    llm: LLM
    ranker: Ranker

    @retry(
        stop=stop_after_attempt(DEFAULT_NUM_MAX_TRY),
        after=after_log(logger, logging.WARNING),
        reraise=True,
    )
    def _generate_layout(
        self, prompt_messages: List[Dict[str, str]], **kwargs
    ) -> List[RankerOutput]:
        try:
            response = self.llm(prompt_messages, **kwargs)
            return self.ranker(response)
        except Exception as err:
            logger.exception(err)
            raise err

    def get_exemplars(
        self, test_data: ProcessedLayoutData
    ) -> List[ProcessedLayoutData]:
        return self.selector(test_data)

    def build_prompt_messages(
        self,
        test_data: ProcessedLayoutData,
        exemplars: Optional[List[ProcessedLayoutData]] = None,
    ) -> List[Dict[str, str]]:
        prompt = self.serializer.build_prompt(
            exemplars=(
                exemplars if exemplars is not None else self.get_exemplars(test_data)
            ),
            layout_data=test_data,
        )
        prompt_messages = [
            {"role": "system", "content": prompt["system_prompt"]},
            {"role": "user", "content": prompt["user_prompt"]},
        ]
        return prompt_messages

    def generate_layout(
        self, prompt_messages: List[Dict[str, str]], max_num_try: int = 5, **kwargs
    ) -> List[RankerOutput]:
        try:
            retry_layout_generator = self._generate_layout.retry_with(
                stop=stop_after_attempt(max_num_try)
            )
            return retry_layout_generator(self, prompt_messages, **kwargs)
        except Exception as err:
            raise LayoutPrompterException(
                f"Failed to generate layout for prompt: {prompt_messages}"
            ) from err

    def __call__(
        self,
        test_data: ProcessedLayoutData,
        max_num_try: int = 5,
        exemplars: Optional[List[ProcessedLayoutData]] = None,
        **kwargs,
    ) -> List[RankerOutput]:
        prompt_messages = self.build_prompt_messages(
            test_data=test_data, exemplars=exemplars
        )
        return self.generate_layout(prompt_messages, max_num_try=max_num_try, **kwargs)
