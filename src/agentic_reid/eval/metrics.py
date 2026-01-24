from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class RetrievalResult:
    ranked_rel: List[str]  # ranked list of rel_paths
    target_rel: str

def rank_of_target(ranked: List[str], target: str) -> Optional[int]:
    try:
        return ranked.index(target) + 1  # 1-based rank
    except ValueError:
        return None

def rank_at_k(rank: Optional[int], k: int) -> int:
    return int(rank is not None and rank <= k)

def average_turns_to_success(turn_ranks: List[Optional[int]], k: int = 1) -> float:
    """
    ATS@k: first turn where rank<=k; if never, return T+1 (penalty).
    """
    for t, r in enumerate(turn_ranks, start=1):
        if r is not None and r <= k:
            return float(t)
    return float(len(turn_ranks) + 1)
