from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def _norm(p: str) -> str:
    return os.path.normpath(p).replace("\\", "/")


def list_images_with_rel(root: str) -> List[Tuple[str, str]]:
    """
    Recursively list images under `root`.
    Returns (rel_path, abs_path), rel_path is relative to root, normalized with '/'.
    """
    items: List[Tuple[str, str]] = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(IMG_EXTS):
                abs_path = os.path.join(dirpath, fn)
                rel_path = _norm(os.path.relpath(abs_path, root))
                items.append((rel_path, abs_path))
    items.sort(key=lambda x: x[0])
    return items


def load_invalid_paths(csv_path: str) -> set[str]:
    """
    invalid_paths.csv can be a single-column CSV containing paths (relative or with split prefix).
    We'll normalize everything to '/'.
    """
    if not os.path.exists(csv_path):
        return set()

    bad = set()
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            bad.add(_norm(row[0].strip()))
    return bad


def _infer_cols(fieldnames: List[str]) -> Tuple[str, str]:
    """
    Try to infer image and caption column names.
    """
    lower = [c.lower() for c in fieldnames]
    img_candidates = {"image", "img", "image_path", "path", "file", "filename", "file_name"}
    cap_candidates = {"caption", "text", "description", "query", "sentence", "sent"}

    img_col = None
    cap_col = None
    for c, lc in zip(fieldnames, lower):
        if lc in img_candidates and img_col is None:
            img_col = c
        if lc in cap_candidates and cap_col is None:
            cap_col = c

    if img_col is None or cap_col is None:
        raise ValueError(f"Could not infer CSV columns. Found columns: {fieldnames}")
    return img_col, cap_col


def load_captions_csv(csv_path: str, split: str) -> Dict[str, List[str]]:
    """
    Returns mapping: rel_path (relative to imgs/<split>) -> list of captions.

    Works for common CSV formats. If CSV stores 'test/xxx.jpg', we strip 'test/'.
    If CSV stores 'imgs/test/xxx.jpg', we strip up to '<split>/'.
    """
    if not os.path.exists(csv_path):
        return {}

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        img_col, cap_col = _infer_cols(fieldnames)

        mapping: Dict[str, List[str]] = {}
        split_prefix = f"{split}/"

        for row in reader:
            raw_path = _norm(row[img_col].strip())
            cap = (row[cap_col] or "").strip()
            if not raw_path or not cap:
                continue

            # Normalize to rel_path under imgs/<split>
            p = raw_path

            # Remove any leading directories until we see "<split>/"
            if split_prefix in p:
                p = p.split(split_prefix, 1)[1]
            # If it is already "0001/abc.jpg" we keep it

            p = _norm(p)
            mapping.setdefault(p, []).append(cap)

        return mapping


@dataclass
class ICFGItem:
    rel_path: str
    abs_path: str
    captions: Optional[List[str]] = None


@dataclass
class ICFGSplit:
    split: str
    root_dir: str                 # .../ICFG-PEDES/imgs/<split>
    items: List[ICFGItem]


def load_icfg_pedes_split(
    dataset_root: str,            # .../ICFG-PEDES
    split: str = "test",
    captions_file: str = "captions_cleaned.csv",  # or "captions.csv" or "" for none
    invalid_paths_file: str = "invalid_paths.csv",
) -> ICFGSplit:
    """
    Loads ICFG-PEDES split from:
      dataset_root/imgs/<split>/** (recursive)

    Optionally attaches captions from captions_cleaned.csv or captions.csv at dataset_root.
    Optionally filters invalid_paths.csv.
    """
    split_root = os.path.join(dataset_root, "imgs", split)
    if not os.path.isdir(split_root):
        raise FileNotFoundError(f"Missing split folder: {split_root}")

    # Index images
    rel_abs = list_images_with_rel(split_root)
    if len(rel_abs) == 0:
        raise ValueError(f"No images found under: {split_root}")

    # Invalid paths
    bad = load_invalid_paths(os.path.join(dataset_root, invalid_paths_file))
    # normalize: bad may include "test/xxx" so we strip split prefix if present
    bad_rel = set()
    for b in bad:
        if b.startswith(f"{split}/"):
            bad_rel.add(_norm(b[len(f"{split}/"):]))
        elif f"{split}/" in b:
            bad_rel.add(_norm(b.split(f"{split}/", 1)[1]))
        else:
            bad_rel.add(_norm(b))

    # Captions
    caps_by_rel: Dict[str, List[str]] = {}
    if captions_file:
        csv_path = os.path.join(dataset_root, captions_file)
        if os.path.exists(csv_path):
            caps_by_rel = load_captions_csv(csv_path, split=split)

    items: List[ICFGItem] = []
    for rel, abs_p in rel_abs:
        if rel in bad_rel:
            continue
        caps = caps_by_rel.get(rel)
        items.append(ICFGItem(rel_path=rel, abs_path=abs_p, captions=caps))

    return ICFGSplit(split=split, root_dir=split_root, items=items)
