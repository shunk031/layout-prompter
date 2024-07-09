from __future__ import annotations

import logging

import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor

logger = logging.getLogger(__name__)


class CLIPTextEncoderTransform(nn.Module):
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32") -> None:
        super().__init__()

        self.model = self._get_clip_model(model_name)
        self.processor = self._get_clip_processor(model_name)

    def _get_clip_model(self, model_name: str) -> CLIPModel:
        model: CLIPModel = CLIPModel.from_pretrained(model_name)  # type: ignore
        model.eval()

        model = model.to("cuda" if torch.cuda.is_available() else "cpu")  # type: ignore
        return model

    def _get_clip_processor(self, model_name: str) -> CLIPProcessor:
        processor: CLIPProcessor = CLIPProcessor.from_pretrained(model_name)  # type: ignore
        return processor

    @torch.no_grad()
    def __call__(self, text: str):
        inputs = self.processor(
            text,
            return_tensors="pt",
            max_length=self.processor.tokenizer.model_max_length,  # type: ignore
            padding="max_length",
            truncation=True,
        )
        text_features = self.model.get_text_features(**inputs)  # type: ignore
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features
