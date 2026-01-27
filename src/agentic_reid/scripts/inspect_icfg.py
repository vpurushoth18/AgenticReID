import argparse
from agentic_reid.data.icfg_pedes import load_icfg_pedes_split

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Path to ICFG-PEDES folder")
    ap.add_argument("--split", default="test")
    ap.add_argument("--captions-file", default="captions_cleaned.csv")
    args = ap.parse_args()

    ds = load_icfg_pedes_split(
        dataset_root=args.root,
        split=args.split,
        captions_file=args.captions_file,
    )

    print(f"Split: {ds.split}")
    print(f"Root:  {ds.root_dir}")
    print(f"Items: {len(ds.items)}")

    with_caps = sum(1 for x in ds.items if x.captions)
    print(f"Items with captions: {with_caps}")

    for ex in ds.items[:3]:
        print("----")
        print("rel:", ex.rel_path)
        print("abs:", ex.abs_path)
        print("caps:", (ex.captions[:2] if ex.captions else None))

if __name__ == "__main__":
    main()

    
