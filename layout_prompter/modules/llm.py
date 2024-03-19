import abc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TypedDict

import requests
from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
from transformers import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer

from layout_prompter.parsers import GPTResponseParser, Parser, TGIResponseParser

__all__ = ["LLM", "GPTCallar", "TGICaller"]


class LLM(object, metaclass=abc.ABCMeta):
    parser: Parser

    @abc.abstractmethod
    def generate(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    def __call__(self, *args, **kwargs) -> Any:
        generated_results = self.generate(*args, **kwargs)
        return self.parser(generated_results)


@dataclass
class GPTCallar(LLM):
    """The GPT caller."""

    parser: GPTResponseParser

    model: str
    max_tokens: int

    temperature: float = 0.7
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    num_return: int = 10
    stop_token: str = "\n\n"

    api_key: Optional[str] = field(repr=False, default=None)
    api_base: Optional[str] = field(repr=False, default=None)

    _client: Optional[OpenAI] = None

    def __post_init__(self) -> None:
        self._client = OpenAI(api_key=self.api_key, base_url=self.api_base)

    @property
    def client(self) -> OpenAI:
        assert self._client is not None
        return self._client

    def generate(
        self, prompt_messages: List[ChatCompletionMessageParam], **kwargs
    ) -> ChatCompletion:
        response = self.client.chat.completions.create(
            model=kwargs["model"] if "model" in kwargs else self.model,
            temperature=(
                kwargs["temperature"] if "temperature" in kwargs else self.temperature
            ),
            max_tokens=(
                kwargs["max_tokens"] if "max_tokens" in kwargs else self.max_tokens
            ),
            top_p=kwargs["top_p"] if "top_p" in kwargs else self.top_p,
            frequency_penalty=(
                kwargs["frequency_penalty"]
                if "frequency_penalty" in kwargs
                else self.frequency_penalty
            ),
            presence_penalty=(
                kwargs["presence_penalty"]
                if "presence_penalty" in kwargs
                else self.presence_penalty
            ),
            n=kwargs["n"] if "n" in kwargs else self.num_return,
            messages=prompt_messages,
            **kwargs,
        )
        return response


class TGIToken(TypedDict):
    id: int
    text: str
    logprob: float
    special: bool


class TGISequence(TypedDict):
    generated_text: str
    finish_reason: str
    generated_tokens: int
    seed: int
    prefill: List[str]
    tokens: List[TGIToken]


class TGIDetails(TypedDict):
    finish_reason: str
    generated_tokens: int
    seed: int
    prefill: List[str]
    tokens: List[TGIToken]
    best_of_sequences: List[TGISequence]


class TGIOutput(TypedDict):
    generated_text: str
    details: TGIDetails


@dataclass
class TGICaller(LLM):
    """The text generation inference (TGI) caller."""

    parser: TGIResponseParser

    endpoint_url: str
    model_id: str

    top_k: int = 50
    top_p: float = 0.95
    do_sample: bool = False

    max_new_tokens: int = 1024
    num_return_sequences: int = 1

    details: bool = True

    _tokenizer: Optional[PreTrainedTokenizer] = None
    _eos_token_id: Optional[int] = None

    def __post_init__(self) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)  # type: ignore
        self._eos_token_id = self.tokenizer.eos_token_id

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        assert self._tokenizer is not None
        return self._tokenizer

    @property
    def eos_token_id(self) -> int:
        assert self._eos_token_id is not None
        return self._eos_token_id

    def generate(self, prompt_messages: List[Dict[str, str]]) -> TGIOutput:
        prompt = self.tokenizer.apply_chat_template(
            prompt_messages, add_generation_prompt=True, tokenize=False
        )
        headers = {"Content-Type": "application/json"}
        params = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "eos_token_id": self.eos_token_id,
            "best_of": self.num_return_sequences,
            "details": self.details,
        }
        data = {
            "inputs": prompt,
            "parameters": params,
        }
        return requests.post(url=self.endpoint_url, headers=headers, json=data).json()
