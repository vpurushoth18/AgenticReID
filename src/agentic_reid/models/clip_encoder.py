from __future__ import annotations

from dataclasses import dataclass
from typing import List, Union

import numpy as np
import torch
from PIL import Image

import clip


@dataclass
class ClipEncoder:
    model_name: str = "ViT-B/16"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self) -> None:
        self.model, self.preprocess = clip.load(self.model_name, device=self.device)
        self.model.eval()

    @torch.no_grad()
    def encode_images(self, image_paths: List[str], batch_size: int = 64) -> torch.Tensor:
        feats: List[torch.Tensor] = []
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            images = torch.stack([self.preprocess(Image.open(p).convert("RGB")) for p in batch_paths])
            images = images.to(self.device)
            emb = self.model.encode_image(images)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            feats.append(emb.detach().cpu())
        return torch.cat(feats, dim=0)

    @torch.no_grad()
    def encode_text(self, texts: Union[str, List[str]]) -> torch.Tensor:
        if isinstance(texts, str):
            texts = [texts]
        tokens = clip.tokenize(texts).to(self.device)
        emb = self.model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.detach().cpu()
