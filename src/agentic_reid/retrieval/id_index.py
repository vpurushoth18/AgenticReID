# id_index.py
import torch
from dataclasses import dataclass
from collections import defaultdict
from typing import Literal

@dataclass
class IdIndex:
    """
    Identity-level index for multi-view person ReID.
    Implements Upgrade 1: aggregates image-level scores to identity-level scores.
    """
    rel_paths: list[str]
    embeddings: torch.Tensor  # [N, D]
    pid_to_indices: dict[str, list[int]]
    
    @staticmethod
    def from_gallery_index(idx: dict) -> "IdIndex":
        """
        Build identity index from gallery.
        Assumes rel_paths format: "person_id/view_xxx.jpg"
        """
        rel = idx["rel_paths"]
        pid_to_idx = defaultdict(list)
        
        for i, p in enumerate(rel):
            pid = p.split("/")[0]
            pid_to_idx[pid].append(i)
        
        return IdIndex(
            rel_paths=rel,
            embeddings=idx["embeddings"],
            pid_to_indices=dict(pid_to_idx)
        )
    
    def score_ids(
        self, 
        image_scores: torch.Tensor, 
        mode: Literal["max", "topm"] = "max", 
        top_m: int = 5
    ) -> tuple[list[str], torch.Tensor]:
        """
        Aggregate image-level scores to identity-level scores.
        
        Args:
            image_scores: [N] tensor of per-image scores
            mode: "max" for max pooling, "topm" for top-m mean pooling
            top_m: number of top views to average (only used if mode="topm")
        
        Returns:
            (person_ids, id_scores): lists of person IDs and their aggregated scores
        """
        out_pids = []
        out_scores = []
        
        for pid, inds in self.pid_to_indices.items():
            if len(inds) == 0:
                continue  # Skip empty identities
            
            s = image_scores[inds]
            
            if mode == "max":
                sc = s.max()
            elif mode == "topm":
                m = min(top_m, s.numel())
                sc = s.topk(m, sorted=True).values.mean()
            else:
                raise ValueError(f"mode must be 'max' or 'topm', got {mode}")
            
            out_pids.append(pid)
            out_scores.append(sc)
        
        return out_pids, torch.stack(out_scores)
    
    def get_top_k_ids(
        self, 
        image_scores: torch.Tensor, 
        k: int = 10, 
        mode: str = "max", 
        top_m: int = 5
    ) -> list[tuple[str, float]]:
        """
        Return top-k person IDs with scores (useful for evaluation).
        
        Returns:
            List of (person_id, score) tuples, sorted by score descending
        """
        pids, scores = self.score_ids(image_scores, mode, top_m)
        ranked = sorted(zip(pids, scores.tolist()), key=lambda x: x[1], reverse=True)
        return ranked[:k]
    
    @property
    def num_identities(self) -> int:
        """Total number of unique person IDs"""
        return len(self.pid_to_indices)
    
    @property
    def num_images(self) -> int:
        """Total number of images across all identities"""
        return len(self.rel_paths)