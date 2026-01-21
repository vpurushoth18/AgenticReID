from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any
import os
import json

import torch
from tqdm import tqdm

from agentic_reid.models.clip_encoder import ClipEncoder


def list_images(root: str) -> List[str]:
    exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    paths = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.lower().endswith(exts):
                paths.append(os.path.join(dirpath, f))
    paths.sort()
    return paths


@dataclass
class GalleryIndex:
    image_paths: List[str]
    embeddings: torch.Tensor  # [N, D], normalized

    def save(self, path: str) -> None:
        payload = {
            "image_paths": self.image_paths,
            "embeddings": self.embeddings,
        }
        torch.save(payload, path)

    @staticmethod
    def load(path: str) -> "GalleryIndex":
        payload = torch.load(path, map_location="cpu")
        return GalleryIndex(
            image_paths=payload["image_paths"],
            embeddings=payload["embeddings"],
        )


def build_index(
    gallery_dir: str,
    out_path: str,
    model_name: str = "ViT-B/16",
    batch_size: int = 64,
) -> None:
    image_paths = list_images(gallery_dir)
    if len(image_paths) == 0:
        raise ValueError(f"No images found under: {gallery_dir}")

    enc = ClipEncoder(model_name=model_name)
    embeddings = enc.encode_images(image_paths, batch_size=batch_size)

    idx = GalleryIndex(image_paths=image_paths, embeddings=embeddings)
    idx.save(out_path)
