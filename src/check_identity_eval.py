import json
import torch
from collections import defaultdict
from agentic_reid.models.clip_encoder import ClipEncoder

# Load index
idx = torch.load("cache/icfg_test_clip_vitb16.pt", map_location="cpu")
emb = idx["embeddings"].float()
rel = idx["rel_paths"]
ids = [p.split("/")[0] for p in rel]

# Load episodes
eps = json.load(open("data/icfg_test_episodes.json"))

enc = ClipEncoder("ViT-B/16")

hit10 = 0

for e in eps:
    target_id = e["target_rel"].split("/")[0]

    q = enc.encode_text(e["init_caption"])[0].float()
    scores = (emb @ q).tolist()

    best_score_per_id = defaultdict(lambda: -1e9)

    for i, score in enumerate(scores):
        pid = ids[i]
        if score > best_score_per_id[pid]:
            best_score_per_id[pid] = score

    ranked_ids = sorted(best_score_per_id.items(), key=lambda x: x[1], reverse=True)
    top10_ids = [pid for pid, _ in ranked_ids[:10]]

    if target_id in top10_ids:
        hit10 += 1

print("Episodes:", len(eps))
print("ID-R@10 (identity-level max pooling):", hit10 / len(eps))
