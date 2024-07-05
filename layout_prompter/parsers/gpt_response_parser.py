from dataclasses import dataclass
from typing import List

from openai.types.chat import ChatCompletion, ChatCompletionMessage

from layout_prompter.parsers import Parser, ParserOutput


@dataclass
class GPTResponseParser(Parser):
    def check_filtered_response_count(
        self, original_response: ChatCompletion, parsed_response: List[ParserOutput]
    ) -> None:
        num_return = len(original_response.choices)
        self.log_filter_response_count(num_return, parsed_response)

    def parse(  # type: ignore[override]
        self,
        response: ChatCompletion,
    ) -> List[ParserOutput]:
        assert isinstance(response, ChatCompletion), type(response)

        parsed_predictions: List[ParserOutput] = []
        for choice in response.choices:
            message = choice.message
            assert isinstance(message, ChatCompletionMessage), type(message)
            content = message.content
            assert content is not None
            parsed_predictions.append(self._extract_labels_and_bboxes(content))

        return parsed_predictions
