"""Lightweight frozen CLIP text encoder for language conditioning."""

from __future__ import annotations

import torch
import torch.nn as nn


class CLIPTextEncoder(nn.Module):
    def __init__(self, model_id: str = "openai/clip-vit-base-patch32", proj_dim: int = 512):
        super().__init__()
        from transformers import CLIPTextModel, CLIPTokenizer

        self.tokenizer = CLIPTokenizer.from_pretrained(model_id)
        self.text_model = CLIPTextModel.from_pretrained(model_id)
        self.text_model.eval()
        for p in self.text_model.parameters():
            p.requires_grad = False
        hidden = self.text_model.config.hidden_size
        self.proj = (
            nn.Linear(hidden, proj_dim) if hidden != proj_dim else nn.Identity()
        )

    @torch.no_grad()
    def forward(self, instructions: list[str]) -> torch.Tensor:
        tokens = self.tokenizer(
            instructions,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        device = next(self.parameters()).device
        tokens = {k: v.to(device) for k, v in tokens.items()}
        out = self.text_model(**tokens).last_hidden_state
        return self.proj(out)
