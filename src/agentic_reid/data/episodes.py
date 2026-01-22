from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import json

@dataclass
class Episode:
    target_rel: str          # e.g. "0000/0000_004_....jpg"
    init_caption: str        # starting description
    all_captions: List[str]  # all available captions for target
    split: str = "test"

def save_episodes(path: str, episodes: List[Episode]) -> None:
    payload = [e.__dict__ for e in episodes]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def load_episodes(path: str) -> List[Episode]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return [Episode(**x) for x in payload]
