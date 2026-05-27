"""Build per-variant YOLO-ready trees for SortWaste.

Why: ultralytics derives label paths from image paths via `realpath` + a
`/images/ -> /labels/` string substitution. If images are *symlinks* to the
raw dataset, realpath redirects to a location where our labels aren't. Plain
symlinks therefore don't work. **Hardlinks** preserve the link path under
realpath, so ultralytics finds labels next to images in our managed tree.

Builds (idempotently):

    data/sortwaste/yolo_<variant>/{train,val,test}/
        images/<stem>.png  -- hardlink to splited_all_dataset/<split>/images/<stem>.png
        labels/<stem>.txt  -- copy of data/sortwaste/labels_<variant>/<split>/<stem>.txt
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

SPLITS = ("train", "val", "test")
VARIANTS = ("4class", "8class")


def link_images(src_dir: Path, dst_dir: Path) -> int:
    dst_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in src_dir.iterdir():
        if p.suffix.lower() != ".png":
            continue
        target = dst_dir / p.name
        src_stat = p.stat()
        if target.exists():
            if target.stat().st_ino == src_stat.st_ino:
                n += 1
                continue
            target.unlink()
        os.link(p, target)
        n += 1
    return n


def copy_labels(src_dir: Path, dst_dir: Path) -> int:
    dst_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in src_dir.iterdir():
        if p.suffix != ".txt":
            continue
        target = dst_dir / p.name
        if target.exists():
            if target.stat().st_size == p.stat().st_size and target.read_bytes() == p.read_bytes():
                n += 1
                continue
            target.unlink()
        shutil.copyfile(p, target)
        n += 1
    return n


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent
    sortwaste = project_root / "data" / "sortwaste"
    raw_root = sortwaste / "sortwaste_raw" / "dataset" / "splited_all_dataset"

    if not raw_root.is_dir():
        sys.exit(f"Missing raw root: {raw_root}")

    print(f"{'variant':8} {'split':5} {'images':>8} {'labels':>8}")
    for variant in VARIANTS:
        labels_src_root = sortwaste / f"labels_{variant}"
        if not labels_src_root.is_dir():
            sys.exit(f"Missing labels source: {labels_src_root}")
        out_root = sortwaste / f"yolo_{variant}"
        for split in SPLITS:
            img_src = raw_root / split / "images"
            lbl_src = labels_src_root / split
            if not img_src.is_dir():
                sys.exit(f"Missing image source: {img_src}")
            if not lbl_src.is_dir():
                sys.exit(f"Missing labels source: {lbl_src}")

            n_img = link_images(img_src, out_root / split / "images")
            n_lbl = copy_labels(lbl_src, out_root / split / "labels")

            print(f"{variant:8} {split:5} {n_img:>8} {n_lbl:>8}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
