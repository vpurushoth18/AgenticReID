import argparse
from agentic_reid.retrieval.index import build_index

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gallery-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="ViT-B/16")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    build_index(args.gallery_dir, args.out, model_name=args.model, batch_size=args.batch_size)
    print(f"Saved index to {args.out}")

if __name__ == "__main__":
    main()
