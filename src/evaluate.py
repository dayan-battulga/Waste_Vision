"""Evaluate a trained YOLOv8 checkpoint on a SortWaste split.

Runs ``model.val()`` on the requested split and emits a per-class + Overall
CSV plus a formatted stdout table in the format of paper Tables 6 (8-class)
and 7 (4-class).
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter
from pathlib import Path

import yaml

SRC_DIR = Path(__file__).resolve().parent

from ultralytics import YOLO  # noqa: E402

VARIANT_NC = {"4class": 4, "8class": 8}
PAPER_OVERALL = {"4class": (0.597, 0.753), "8class": (0.451, 0.567)}


def count_instances(labels_split_dir: Path, nc: int) -> tuple[dict[int, int], int]:
    counts: Counter = Counter()
    for txt in labels_split_dir.iterdir():
        if txt.suffix != ".txt":
            continue
        with txt.open() as f:
            for line in f:
                parts = line.split()
                if not parts:
                    continue
                cid = int(parts[0])
                counts[cid] += 1
    out = {c: counts.get(c, 0) for c in range(nc)}
    return out, sum(out.values())


def fmt(x: float | None, prec: int = 3) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:.{prec}f}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--variant", choices=list(VARIANT_NC), required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--device", type=str, default="mps")
    args = p.parse_args()

    project_root = SRC_DIR.parent
    sortwaste_dir = project_root / "data" / "sortwaste"

    if not args.weights.is_file():
        sys.exit(f"Weights not found: {args.weights}")
    if not args.data.is_file():
        sys.exit(f"Data YAML not found: {args.data}")

    data_yaml_path = args.data.resolve()
    with data_yaml_path.open() as f:
        data_cfg = yaml.safe_load(f)
    names = data_cfg["names"]
    if isinstance(names, list):
        class_names = {i: n for i, n in enumerate(names)}
    else:
        class_names = {int(k): v for k, v in names.items()}
    nc = len(class_names)
    expected_nc = VARIANT_NC[args.variant]
    if nc != expected_nc:
        sys.exit(
            f"YAML nc mismatch: {data_yaml_path} has nc={nc}, "
            f"variant {args.variant!r} expects nc={expected_nc}"
        )

    labels_split_dir = sortwaste_dir / f"yolo_{args.variant}" / args.split / "labels"
    if not labels_split_dir.is_dir():
        sys.exit(f"Labels split dir missing: {labels_split_dir}")
    n_per_class, n_total = count_instances(labels_split_dir, nc)

    eval_project = project_root / "runs" / "eval"
    run_name = args.weights.parent.parent.name + f"_{args.split}"

    model = YOLO(str(args.weights))
    metrics = model.val(
        data=str(data_yaml_path),
        split=args.split,
        imgsz=args.imgsz,
        device=args.device,
        project=str(eval_project),
        name=run_name,
        exist_ok=True,
    )

    box = metrics.box
    ap_class_index = list(box.ap_class_index)

    per_class: dict[int, dict[str, float]] = {}
    for cid in range(nc):
        per_class[cid] = {
            "precision": float("nan"),
            "recall": float("nan"),
            "ap50": float("nan"),
            "ap75": float("nan"),
            "ap": float("nan"),
        }
    for i, cid in enumerate(ap_class_index):
        cid = int(cid)
        per_class[cid]["precision"] = float(box.p[i])
        per_class[cid]["recall"] = float(box.r[i])
        per_class[cid]["ap50"] = float(box.ap50[i])
        try:
            per_class[cid]["ap75"] = float(box.all_ap[i, 5])
        except Exception:
            per_class[cid]["ap75"] = float("nan")
        per_class[cid]["ap"] = float(box.maps[cid])

    overall = {
        "precision": float(box.mp),
        "recall": float(box.mr),
        "ap50": float(box.map50),
        "ap75": float(box.map75),
        "ap": float(box.map),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "class_name", "n_instances",
                         "precision", "recall", "AP50", "AP75", "AP"])

        def cell(x: float) -> str:
            return "" if math.isnan(x) else f"{x:.6f}"

        for cid in range(nc):
            m = per_class[cid]
            writer.writerow([
                cid, class_names[cid], n_per_class[cid],
                cell(m["precision"]), cell(m["recall"]),
                cell(m["ap50"]), cell(m["ap75"]), cell(m["ap"]),
            ])
        writer.writerow([
            "overall", args.variant, n_total,
            cell(overall["precision"]), cell(overall["recall"]),
            cell(overall["ap50"]), cell(overall["ap75"]), cell(overall["ap"]),
        ])

    paper_ap, paper_ap50 = PAPER_OVERALL[args.variant]

    header = ("Class", "n", "P", "R", "AP50", "AP75", "AP", "Paper (AP / AP50)")
    widths = [20, 7, 7, 7, 7, 7, 7, 20]
    print("  ".join(h.ljust(w) for h, w in zip(header, widths)))
    for cid in range(nc):
        m = per_class[cid]
        row = (
            class_names[cid],
            str(n_per_class[cid]),
            fmt(m["precision"]), fmt(m["recall"]),
            fmt(m["ap50"]), fmt(m["ap75"]), fmt(m["ap"]),
            "—",
        )
        print("  ".join(c.ljust(w) for c, w in zip(row, widths)))
    row = (
        "Overall",
        str(n_total),
        fmt(overall["precision"]), fmt(overall["recall"]),
        fmt(overall["ap50"]), fmt(overall["ap75"]), fmt(overall["ap"]),
        f"{paper_ap:.3f} / {paper_ap50:.3f}",
    )
    print("  ".join(c.ljust(w) for c, w in zip(row, widths)))

    zero_classes = [class_names[c] for c in range(nc) if n_per_class[c] == 0]
    if zero_classes:
        print(f"\nNote: classes {zero_classes} have 0 ground-truth instances "
              f"in split={args.split!r}; reported as NaN.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
