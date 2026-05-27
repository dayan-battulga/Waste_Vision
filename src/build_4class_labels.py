"""Derive 4-class plastic-only labels from the released 8-class labels and
diff them against the released 4-class labels.

Like `src/convert_annotations.py`, this is a reproducibility check — both label
sets ship with the dataset; deriving one from the other proves they are
internally consistent.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

SPLITS = ("train", "val", "test")

REMAP: dict[int, int | None] = {
    0: 0,    # PET                 -> PET
    1: 1,    # HDPE                -> HDPE
    2: 2,    # Mixed_Soft_Plastic  -> Mixed_Plastic
    3: 3,    # ECAL                -> ECAL
    4: None, # Metal               -> DROP
    5: None, # Cardboard           -> DROP
    6: 2,    # Mixed_Rigid_Plastic -> Mixed_Plastic
    7: 0,    # PET_Oil             -> PET
}
CLASS_NAMES_4 = ["PET", "HDPE", "Mixed_Plastic", "ECAL"]
COORD_TOL = 1e-6


def remap_file(src_path: Path) -> list[str]:
    """Return derived YOLO lines (each ending with newline)."""
    out_lines: list[str] = []
    with src_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.split(maxsplit=1)
            cls = int(parts[0])
            rest = parts[1] if len(parts) > 1 else "\n"
            new_cls = REMAP[cls]
            if new_cls is None:
                continue
            if not rest.endswith("\n"):
                rest += "\n"
            out_lines.append(f"{new_cls} {rest}")
    return out_lines


def parse_lines(lines: list[str]) -> list[tuple[int, tuple[float, float, float, float]]]:
    boxes: list[tuple[int, tuple[float, float, float, float]]] = []
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        cls = int(parts[0])
        cx, cy, w, h = (float(p) for p in parts[1:5])
        boxes.append((cls, (cx, cy, w, h)))
    return boxes


def diff_boxes(
    derived: list[tuple[int, tuple[float, float, float, float]]],
    released: list[tuple[int, tuple[float, float, float, float]]],
) -> tuple[bool, bool]:
    d_classes = sorted(c for c, _ in derived)
    r_classes = sorted(c for c, _ in released)
    class_ids_match = d_classes == r_classes

    if len(derived) != len(released):
        return class_ids_match, False

    d_sorted = sorted(derived, key=lambda x: (x[0], *x[1]))
    r_sorted = sorted(released, key=lambda x: (x[0], *x[1]))
    coords_match = True
    for (dc, dcoords), (rc, rcoords) in zip(d_sorted, r_sorted):
        if dc != rc:
            coords_match = False
            break
        for a, b in zip(dcoords, rcoords):
            if abs(a - b) > COORD_TOL:
                coords_match = False
                break
        if not coords_match:
            break
    return class_ids_match, coords_match


def process_split(
    split: str,
    src_labels: Path,
    out_labels: Path,
    released_4class: Path,
) -> tuple[dict, list[dict], Counter, Counter, list[str]]:
    src_dir = src_labels / split
    out_dir = out_labels / split
    rel_dir = released_4class / split
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    derived_counts: Counter = Counter()
    released_counts: Counter = Counter()
    mismatches: list[str] = []
    total_derived = 0
    total_released = 0
    n_images = 0

    for src_path in sorted(src_dir.iterdir()):
        if src_path.suffix != ".txt":
            continue
        n_images += 1
        stem = src_path.stem

        derived_lines = remap_file(src_path)
        out_path = out_dir / src_path.name
        with out_path.open("w") as f:
            f.writelines(derived_lines)

        derived_boxes = parse_lines(derived_lines)
        rel_path = rel_dir / src_path.name
        if rel_path.exists():
            with rel_path.open() as f:
                released_lines = f.readlines()
            released_boxes = parse_lines(released_lines)
        else:
            released_boxes = []

        for cls, _ in derived_boxes:
            derived_counts[cls] += 1
        for cls, _ in released_boxes:
            released_counts[cls] += 1
        total_derived += len(derived_boxes)
        total_released += len(released_boxes)

        if not rel_path.exists():
            class_ids_match = len(derived_boxes) == 0
            coords_match = len(derived_boxes) == 0
        else:
            class_ids_match, coords_match = diff_boxes(derived_boxes, released_boxes)

        if not class_ids_match or not coords_match:
            mismatches.append(stem)

        rows.append(
            {
                "split": split,
                "stem": stem,
                "n_derived": len(derived_boxes),
                "n_released": len(released_boxes),
                "class_ids_match": class_ids_match,
                "coords_match": coords_match,
            }
        )

    summary = {
        "split": split,
        "images": n_images,
        "boxes_derived": total_derived,
        "boxes_released": total_released,
        "mismatches": len(mismatches),
    }
    return summary, rows, derived_counts, released_counts, mismatches


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src-labels", type=Path, required=True)
    p.add_argument("--out-labels", type=Path, required=True)
    p.add_argument("--released-4class", type=Path, required=True)
    p.add_argument("--diff-report", type=Path, required=True)
    args = p.parse_args()

    args.diff_report.parent.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    summaries: list[dict] = []
    total_derived: Counter = Counter()
    total_released: Counter = Counter()
    mismatches_by_split: dict[str, list[str]] = {}

    for split in SPLITS:
        summary, rows, d_counts, r_counts, mismatches = process_split(
            split, args.src_labels, args.out_labels, args.released_4class
        )
        summaries.append(summary)
        all_rows.extend(rows)
        total_derived.update(d_counts)
        total_released.update(r_counts)
        mismatches_by_split[split] = mismatches

    with args.diff_report.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["split", "stem", "n_derived", "n_released",
                        "class_ids_match", "coords_match"],
        )
        writer.writeheader()
        writer.writerows(all_rows)

    header = ("Split", "Images", "Derived boxes", "Released boxes", "Mismatches")
    widths = [7, 8, 15, 16, 11]
    print("  ".join(h.ljust(w) for h, w in zip(header, widths)))
    overall_ok = True
    for s in summaries:
        if s["mismatches"] > 0:
            overall_ok = False
        row = (s["split"], str(s["images"]), str(s["boxes_derived"]),
               str(s["boxes_released"]), str(s["mismatches"]))
        print("  ".join(c.ljust(w) for c, w in zip(row, widths)))

    print()
    cls_header = ("Class", "Derived", "Released", "Diff")
    cls_widths = [15, 10, 10, 8]
    print("  ".join(h.ljust(w) for h, w in zip(cls_header, cls_widths)))
    for cid, name in enumerate(CLASS_NAMES_4):
        d = total_derived.get(cid, 0)
        r = total_released.get(cid, 0)
        diff = d - r
        if diff != 0:
            overall_ok = False
        row = (name, str(d), str(r), str(diff))
        print("  ".join(c.ljust(w) for c, w in zip(row, cls_widths)))

    for split, ms in mismatches_by_split.items():
        if ms:
            print(f"\n{split} first mismatching stems: {ms[:5]}")

    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
