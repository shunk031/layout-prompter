from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

from layout_prompter.parsers import Parser, ParserOutput

if TYPE_CHECKING:
    from layout_prompter.modules.llm import TGIOutput


@dataclass
class TGIResponseParser(Parser):
    def check_filtered_response_count(
        self, original_response: TGIOutput, parsed_response: List[ParserOutput]
    ) -> None:
        num_return = 1
        num_return += len(original_response["details"]["best_of_sequences"])
        self.log_filter_response_count(num_return, parsed_response)

    def parse(  # type: ignore[override]
        self,
        response: TGIOutput,
    ) -> List[ParserOutput]:
        generated_texts = [response["generated_text"]] + [
            res["generated_text"] for res in response["details"]["best_of_sequences"]
        ]
        parsed_predictions: List[ParserOutput] = []
        for generated_text in generated_texts:
            parsed_predictions.append(self._extract_labels_and_bboxes(generated_text))

        return parsed_predictions
