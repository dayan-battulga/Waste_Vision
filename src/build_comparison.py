"""Build the head-to-head comparison between our YOLOv8m sweep and paper YOLOv11.

Reads test-eval CSVs from ``results/`` and ``run_meta.json`` from
``runs/train/<run>/train/`` (or any depth — globbed). Writes
``results/comparison_sweep_4class.csv`` and ``results/comparison_4class.md``,
and prints both Markdown tables to stdout.

All numbers are parsed from artifacts — none are hand-typed except the paper
reference values (Table 7, YOLOv11 column).
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
RUNS_TRAIN = ROOT / "runs" / "train"

RUNS = [
    "yolov8m_4class_adamw_5e-04",
    "yolov8m_4class_adamw_1e-03",
    "yolov8m_4class_sgd_1e-02",
]

EXPECTED_WINNER = "yolov8m_4class_adamw_5e-04"

PAPER_YOLOV11 = {
    "PET": 0.872,
    "HDPE": 0.729,
    "Mixed_Plastic": 0.624,
    "ECAL": 0.786,
    "Overall_AP50": 0.753,
    "Overall_AP": 0.597,
}

CLASS_ORDER = ["PET", "HDPE", "Mixed_Plastic", "ECAL"]


def find_eval_csv(run: str) -> Path:
    candidates = list(RESULTS.glob(f"eval_{run}.csv"))
    if not candidates:
        raise FileNotFoundError(f"No eval CSV found for run {run}")
    return candidates[0]


def find_run_meta(run: str) -> Path:
    hits = list((RUNS_TRAIN / run).rglob("run_meta.json"))
    if not hits:
        raise FileNotFoundError(f"No run_meta.json under runs/train/{run}/")
    return hits[0]


def parse_eval_csv(path: Path) -> dict:
    per_class: dict[str, dict[str, float]] = {}
    overall: dict[str, float] = {}
    with path.open() as fh:
        for row in csv.DictReader(fh):
            cname = row["class_name"]
            entry = {
                "AP50": float(row["AP50"]),
                "AP": float(row["AP"]),
            }
            if row["class_id"] == "overall":
                overall = entry
            else:
                per_class[cname] = entry
    return {"per_class": per_class, "overall": overall}


def parse_run_meta(path: Path) -> dict:
    meta = json.loads(path.read_text())
    return {
        "optimizer": meta["cli_args"]["optimizer"],
        "lr0": float(meta["cli_args"]["lr0"]),
        "val_mAP50_95": float(meta["best_mAP50_95"]),
        "val_mAP50": float(meta["best_mAP50"]),
    }


def build_rows() -> list[dict]:
    rows = []
    for run in RUNS:
        csv_path = find_eval_csv(run)
        meta_path = find_run_meta(run)
        print(f"  found  {csv_path.relative_to(ROOT)}")
        print(f"  found  {meta_path.relative_to(ROOT)}")
        ev = parse_eval_csv(csv_path)
        meta = parse_run_meta(meta_path)
        rows.append({
            "run": run,
            "optimizer": meta["optimizer"],
            "lr0": meta["lr0"],
            "val_mAP50": meta["val_mAP50"],
            "val_mAP50_95": meta["val_mAP50_95"],
            "test_AP50": ev["overall"]["AP50"],
            "test_AP": ev["overall"]["AP"],
            "per_class_AP50": {k: v["AP50"] for k, v in ev["per_class"].items()},
        })
    winner_idx = max(range(len(rows)), key=lambda i: rows[i]["val_mAP50_95"])
    for i, r in enumerate(rows):
        r["is_winner"] = i == winner_idx
    return rows


def write_sweep_csv(rows: list[dict], path: Path) -> None:
    fieldnames = ["run", "optimizer", "lr0", "val_mAP50", "val_mAP50_95",
                  "test_AP50", "test_AP", "is_winner"]
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fieldnames})


def fmt_lr(lr: float) -> str:
    return f"{lr:.0e}"


def render_table_a(rows: list[dict]) -> str:
    sorted_rows = sorted(rows, key=lambda r: -r["val_mAP50_95"])
    lines = [
        "**Table A — hyperparameter sweep** (sorted by val_mAP50_95 desc)",
        "",
        "| Run | Optimizer | LR | Val AP50-95 | Test AP50 | Test AP |",
        "|---|---|---|---|---|---|",
    ]
    for r in sorted_rows:
        mark = " ✓" if r["is_winner"] else ""
        lines.append(
            f"| {r['run']}{mark} | {r['optimizer']} | {fmt_lr(r['lr0'])} "
            f"| {r['val_mAP50_95']:.3f} | {r['test_AP50']:.3f} | {r['test_AP']:.3f} |"
        )
    return "\n".join(lines)


def render_table_b(winner: dict) -> str:
    lines = [
        "**Table B — winner vs paper, per class**",
        "",
        "| Class | YOLOv8m (ours) AP50 | YOLOv11 (paper) AP50 | Δ |",
        "|---|---|---|---|",
    ]
    for cname in CLASS_ORDER:
        ours = winner["per_class_AP50"][cname]
        paper = PAPER_YOLOV11[cname]
        delta = ours - paper
        lines.append(f"| {cname} | {ours:.3f} | {paper:.3f} | {delta:+.3f} |")
    overall_ap50_ours = winner["test_AP50"]
    overall_ap50_paper = PAPER_YOLOV11["Overall_AP50"]
    overall_ap_ours = winner["test_AP"]
    overall_ap_paper = PAPER_YOLOV11["Overall_AP"]
    lines.append(
        f"| Overall AP50 | {overall_ap50_ours:.3f} | {overall_ap50_paper:.3f} "
        f"| {overall_ap50_ours - overall_ap50_paper:+.3f} |"
    )
    lines.append(
        f"| Overall AP(50-95) | {overall_ap_ours:.3f} | {overall_ap_paper:.3f} "
        f"| {overall_ap_ours - overall_ap_paper:+.3f} |"
    )
    return "\n".join(lines)


def main() -> int:
    print("Reading inputs...")
    rows = build_rows()
    winner = next(r for r in rows if r["is_winner"])

    if winner["run"] != EXPECTED_WINNER:
        print(
            f"\nWARNING: winner by val_mAP50_95 is {winner['run']!r}, "
            f"expected {EXPECTED_WINNER!r}. This contradicts the established "
            f"result — a file is likely being misread. Aborting.",
            file=sys.stderr,
        )
        return 1

    sweep_path = RESULTS / "comparison_sweep_4class.csv"
    write_sweep_csv(rows, sweep_path)
    print(f"\nWrote {sweep_path.relative_to(ROOT)}")

    table_a = render_table_a(rows)
    table_b = render_table_b(winner)
    md = f"# 4-class comparison: YOLOv8m vs paper YOLOv11\n\n{table_a}\n\n{table_b}\n"

    md_path = RESULTS / "comparison_4class.md"
    md_path.write_text(md)
    print(f"Wrote {md_path.relative_to(ROOT)}\n")

    print(table_a)
    print()
    print(table_b)
    return 0


if __name__ == "__main__":
    sys.exit(main())
