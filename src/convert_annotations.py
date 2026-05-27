"""Verify that the released YOLO labels match the source COCO JSON.

Re-derives YOLO boxes from `<split>_plastic.json` in memory and diffs them
against `splited_dataset_plastic_yolo/<split>/labels/<stem>.txt`. Does not
write or modify any label files.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

SPLITS = ("train", "val", "test")

COCO_NAME_TO_YOLO_ID = {
    "pet": 0,
    "pead": 1,
    "mixed_plastic": 2,
    "ecal": 3,
}


def build_cat_map(categories: list[dict]) -> dict[int, int]:
    cat_map: dict[int, int] = {}
    for c in categories:
        name = c["name"].strip().lower()
        if name not in COCO_NAME_TO_YOLO_ID:
            print(f"Unmapped COCO category: id={c['id']} name={c['name']!r}")
            sys.exit(1)
        cat_map[c["id"]] = COCO_NAME_TO_YOLO_ID[name]
    return cat_map


def coco_to_yolo_box(bbox: list[float], img_w: int, img_h: int) -> tuple[float, float, float, float]:
    x, y, w, h = bbox
    return (
        (x + w / 2.0) / img_w,
        (y + h / 2.0) / img_h,
        w / img_w,
        h / img_h,
    )


def parse_yolo_txt(path: Path) -> list[tuple[int, float, float, float, float]]:
    boxes: list[tuple[int, float, float, float, float]] = []
    with path.open() as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            cls = int(parts[0])
            cx, cy, w, h = (float(p) for p in parts[1:5])
            boxes.append((cls, cx, cy, w, h))
    return boxes


def compare_boxes(
    coco: list[tuple[int, float, float, float, float]],
    yolo: list[tuple[int, float, float, float, float]],
    tol: float,
) -> tuple[str, float | None]:
    if len(coco) != len(yolo):
        return "MISMATCH", None
    if not coco:
        return "OK", 0.0
    c_sorted = sorted(coco)
    y_sorted = sorted(yolo)
    max_diff = 0.0
    for a, b in zip(c_sorted, y_sorted):
        if a[0] != b[0]:
            return "MISMATCH", None
        for ai, bi in zip(a[1:], b[1:]):
            d = abs(ai - bi)
            if d > max_diff:
                max_diff = d
    return ("OK" if max_diff <= tol else "MISMATCH", max_diff)


def process_split(
    split: str,
    coco_root: Path,
    yolo_root: Path,
    results_dir: Path,
    tol: float,
) -> dict:
    coco_path = coco_root / f"{split}_plastic.json"
    with coco_path.open() as f:
        data = json.load(f)

    cat_map = build_cat_map(data["categories"])

    anns_by_img: dict[int, list[dict]] = defaultdict(list)
    for ann in data["annotations"]:
        anns_by_img[ann["image_id"]].append(ann)

    labels_dir = yolo_root / split / "labels"
    rows = []
    total_coco = 0
    total_yolo = 0
    mismatched: list[str] = []

    for img in data["images"]:
        stem = Path(img["file_name"]).stem
        w, h = img["width"], img["height"]
        coco_boxes: list[tuple[int, float, float, float, float]] = []
        for ann in anns_by_img.get(img["id"], []):
            cls = cat_map[ann["category_id"]]
            cx, cy, bw, bh = coco_to_yolo_box(ann["bbox"], w, h)
            coco_boxes.append((cls, cx, cy, bw, bh))

        txt_path = labels_dir / f"{stem}.txt"
        if txt_path.exists():
            yolo_boxes = parse_yolo_txt(txt_path)
        else:
            yolo_boxes = []

        status, max_diff = compare_boxes(coco_boxes, yolo_boxes, tol)
        if not txt_path.exists() and coco_boxes:
            status = "MISMATCH"
            max_diff = None

        total_coco += len(coco_boxes)
        total_yolo += len(yolo_boxes)
        if status == "MISMATCH":
            mismatched.append(stem)

        rows.append(
            {
                "stem": stem,
                "n_coco": len(coco_boxes),
                "n_yolo": len(yolo_boxes),
                "max_coord_diff": "" if max_diff is None else f"{max_diff:.6g}",
                "status": status,
            }
        )

    csv_path = results_dir / f"coco_yolo_diff_{split}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["stem", "n_coco", "n_yolo", "max_coord_diff", "status"]
        )
        writer.writeheader()
        writer.writerows(rows)

    return {
        "split": split,
        "images": len(data["images"]),
        "boxes_coco": total_coco,
        "boxes_yolo": total_yolo,
        "mismatched": len(mismatched),
        "mismatched_stems": mismatched[:5],
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--coco-root", type=Path, required=True)
    p.add_argument("--yolo-root", type=Path, required=True)
    p.add_argument("--image-root", type=Path, required=False, default=None,
                   help="Accepted for CLI compatibility; not used (we read width/height from COCO).")
    p.add_argument("--tolerance", type=float, default=1e-4)
    p.add_argument("--results-dir", type=Path,
                   default=Path(__file__).resolve().parent.parent / "results")
    args = p.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for split in SPLITS:
        summaries.append(
            process_split(split, args.coco_root, args.yolo_root, args.results_dir, args.tolerance)
        )

    header = ("Split", "Images", "Boxes(COCO)", "Boxes(YOLO)", "Mismatched", "Status")
    widths = [7, 8, 13, 13, 12, 8]
    print("  ".join(h.ljust(w) for h, w in zip(header, widths)))
    overall_ok = True
    for s in summaries:
        status = "OK" if s["mismatched"] == 0 else "MISMATCH"
        if status != "OK":
            overall_ok = False
        row = (
            s["split"],
            str(s["images"]),
            str(s["boxes_coco"]),
            str(s["boxes_yolo"]),
            str(s["mismatched"]),
            status,
        )
        print("  ".join(c.ljust(w) for c, w in zip(row, widths)))

    for s in summaries:
        if s["mismatched_stems"]:
            print(f"\n{s['split']} first mismatching stems: {s['mismatched_stems']}")

    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
