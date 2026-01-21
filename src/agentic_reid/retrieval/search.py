from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch

from agentic_reid.models.clip_encoder import ClipEncoder
from agentic_reid.retrieval.index import GalleryIndex


@dataclass
class Retriever:
    index: GalleryIndex
    encoder: ClipEncoder

    @torch.no_grad()
    def search(self, query: str, top_k: int = 20) -> List[Tuple[str, float]]:
        q = self.encoder.encode_text(query)  # [1, D] on CPU
        sims = (self.index.embeddings @ q[0])  # [N]
        vals, idxs = torch.topk(sims, k=min(top_k, sims.shape[0]))
        out = [(self.index.image_paths[i], float(v)) for i, v in zip(idxs.tolist(), vals.tolist())]
        return out
