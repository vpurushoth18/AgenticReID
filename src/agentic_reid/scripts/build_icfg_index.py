import argparse
from agentic_reid.retrieval.index import build_index

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--icfg-root", required=True, help="Path to ICFG-PEDES folder")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="cache/icfg_test_clip_vitb16.pt")
    ap.add_argument("--model", default="ViT-B/16")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    gallery_dir = f"{args.icfg_root}/imgs/{args.split}"
    build_index(gallery_dir, args.out, model_name=args.model, batch_size=args.batch_size)
    print(f"Saved index -> {args.out}")

if __name__ == "__main__":
    main()
