import argparse
import json
import torch

from agentic_reid.models.clip_encoder import ClipEncoder
from agentic_reid.retrieval.id_index import IdIndex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--pooling", default="max", choices=["max", "topm"])
    ap.add_argument("--top-m", type=int, default=5)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--model", default="ViT-B/16")
    args = ap.parse_args()

    idx = torch.load(args.index, map_location="cpu")
    id_index = IdIndex.from_gallery_index(idx)
    emb = id_index.embeddings.float()

    eps = json.load(open(args.episodes))
    enc = ClipEncoder(args.model)

    r1 = 0
    rk = 0
    ranks = []

    for e in eps:
        target_pid = e["target_rel"].split("/")[0]
        q = enc.encode_text(e["init_caption"])[0].float()

        image_scores = emb @ q

        pids, pid_scores = id_index.score_ids(
            image_scores,
            mode=args.pooling,
            top_m=args.top_m,
        )

        order = torch.argsort(pid_scores, descending=True)
        ranked_pids = [pids[i] for i in order.tolist()]

        rank = ranked_pids.index(target_pid) + 1
        ranks.append(rank)

        r1 += int(rank == 1)
        rk += int(rank <= args.k)

    n = len(eps)
    ranks_sorted = sorted(ranks)

    print("Episodes:", n)
    print(f"ID-R@1: {r1 / n:.4f}")
    print(f"ID-R@{args.k}: {rk / n:.4f}")
    print(f"Median Rank: {ranks_sorted[n//2]}")
    print("Best Rank:", min(ranks))
    print("Worst Rank:", max(ranks))


if __name__ == "__main__":
    main()
