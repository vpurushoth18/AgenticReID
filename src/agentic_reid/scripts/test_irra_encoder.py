import argparse
import glob
import numpy as np

from agentic_reid.models.irra_adapter import IRRAAdapter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--irra-root", default="externals/IRRA")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--image-glob", required=True)
    ap.add_argument("--text", default="a man wearing a black coat")
    args = ap.parse_args()

    enc = IRRAAdapter(args.irra_root, args.ckpt)

    paths = sorted(glob.glob(args.image_glob))
    if not paths:
        raise SystemExit("No images found. Check your --image-glob")

    sample = paths[0]
    img_feat = enc.encode_images([sample], batch_size=1)
    txt_feat = enc.encode_text(args.text)

    print("Sample image:", sample)
    print("img_feat:", tuple(img_feat.shape))
    print("txt_feat:", tuple(txt_feat.shape))
    print("cosine(sim):", float((img_feat[0] * txt_feat[0]).sum().item()))

    # distribution over first 200 images
    gal_paths = paths[:200]
    gal = enc.encode_images(gal_paths, batch_size=64)
    sims = (gal @ txt_feat[0]).numpy()
    print("gallery:", len(gal_paths))
    print("sim min/mean/max:", float(sims.min()), float(sims.mean()), float(sims.max()))
    top = sims.argsort()[-5:][::-1]
    print("top-5:")
    for i in top:
        print(" ", gal_paths[int(i)], float(sims[int(i)]))


if __name__ == "__main__":
    main()
