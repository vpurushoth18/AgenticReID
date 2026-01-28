import argparse
import numpy as np
import torch

from agentic_reid.data.episodes import load_episodes
from agentic_reid.models.clip_encoder import ClipEncoder
from agentic_reid.retrieval.index import GalleryIndex
from agentic_reid.eval.metrics import rank_of_target, rank_at_k

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True, help=".pt gallery index")
    ap.add_argument("--episodes", required=True, help="episodes json")
    ap.add_argument("--topk", type=int, default=1000)
    ap.add_argument("--model", default="ViT-B/16")
    args = ap.parse_args()

    idx = GalleryIndex.load(args.index)
    enc = ClipEncoder(model_name=args.model)

    eps = load_episodes(args.episodes)

    r1 = r5 = r10 = 0
    ranks = []

    # Preload embeddings on CPU; for speed you can move to GPU later
    gallery = idx.embeddings  # [N, D]

    for e in eps:
        q = enc.encode_text(e.init_caption)[0]  # [D] on CPU
        sims = gallery @ q
        vals, ids = torch.topk(sims, k=min(args.topk, sims.shape[0]))
        ranked_rel = [idx.rel_paths[i] for i in ids.tolist()]

        r = rank_of_target(ranked_rel, e.target_rel)
        ranks.append(r if r is not None else 10**9)

        r1 += rank_at_k(r, 1)
        r5 += rank_at_k(r, 5)
        r10 += rank_at_k(r, 10)

        


    n = len(eps)
    print(f"Episodes: {n}")
    print(f"R@1 : {r1/n:.4f}")
    print(f"R@5 : {r5/n:.4f}")
    print(f"R@10: {r10/n:.4f}")
    print(f"Median rank (capped): {int(np.median(ranks))}")

if __name__ == "__main__":
    main()
