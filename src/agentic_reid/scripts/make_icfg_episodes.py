import argparse
import random
from agentic_reid.data.icfg_pedes import load_icfg_pedes_split
from agentic_reid.data.episodes import Episode, save_episodes

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--icfg-root", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--captions-file", default="captions_cleaned.csv")
    ap.add_argument("--out", default="data/icfg_test_episodes.json")
    ap.add_argument("--max", type=int, default=2000, help="number of episodes to sample")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ds = load_icfg_pedes_split(
        dataset_root=args.icfg_root,
        split=args.split,
        captions_file=args.captions_file,
    )

    rng = random.Random(args.seed)

    # only keep items with captions
    items = [x for x in ds.items if x.captions and len(x.captions) > 0]
    rng.shuffle(items)
    items = items[: min(args.max, len(items))]

    episodes = []
    for it in items:
        # pick one caption as init
        init = rng.choice(it.captions)
        episodes.append(Episode(
            target_rel=it.rel_path,
            init_caption=init,
            all_captions=it.captions,
            split=args.split
        ))

    save_episodes(args.out, episodes)
    print(f"Saved {len(episodes)} episodes -> {args.out}")

if __name__ == "__main__":
    main()
