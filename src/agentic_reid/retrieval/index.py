from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch

from agentic_reid.models.clip_encoder import ClipEncoder
from agentic_reid.data.icfg_pedes import list_images_with_rel


@dataclass
class GalleryIndex:
    root_dir: str
    rel_paths: List[str]
    abs_paths: List[str]
    embeddings: torch.Tensor  # [N, D], normalized

    def save(self, path: str) -> None:
        torch.save(
            {
                "root_dir": self.root_dir,
                "rel_paths": self.rel_paths,
                "abs_paths": self.abs_paths,
                "embeddings": self.embeddings,
            },
            path,
        )

    @staticmethod
    def load(path: str) -> "GalleryIndex":
        payload = torch.load(path, map_location="cpu")
        return GalleryIndex(
            root_dir=payload["root_dir"],
            rel_paths=payload["rel_paths"],
            abs_paths=payload["abs_paths"],
            embeddings=payload["embeddings"],
        )


def build_index(
    gallery_dir: str,             # e.g. .../ICFG-PEDES/imgs/test
    out_path: str,
    model_name: str = "ViT-B/16",
    batch_size: int = 64,
) -> None:
    items = list_images_with_rel(gallery_dir)
    if len(items) == 0:
        raise ValueError(f"No images found under: {gallery_dir}")

    rel_paths = [rp for rp, _ in items]
    abs_paths = [ap for _, ap in items]

    enc = ClipEncoder(model_name=model_name)
    embeddings = enc.encode_images(abs_paths, batch_size=batch_size)

    idx = GalleryIndex(
        root_dir=gallery_dir,
        rel_paths=rel_paths,
        abs_paths=abs_paths,
        embeddings=embeddings,
    )
    idx.save(out_path)
