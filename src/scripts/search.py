import argparse
from agentic_reid.models.clip_encoder import ClipEncoder
from agentic_reid.retrieval.index import GalleryIndex
from agentic_reid.retrieval.search import Retriever

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--query", required=True)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--model", default="ViT-B/16")
    args = ap.parse_args()

    idx = GalleryIndex.load(args.index)
    enc = ClipEncoder(model_name=args.model)
    retr = Retriever(index=idx, encoder=enc)

    results = retr.search(args.query, top_k=args.top_k)
    for p, s in results:
        print(f"{s: .4f}  {p}")

if __name__ == "__main__":
    main()
