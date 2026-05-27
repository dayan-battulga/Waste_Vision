"""Train YOLOv8 on SortWaste under the locked paper-protocol config.

Exposes only the knobs we sweep (model, optimizer, lr, imgsz); flips the
`data/sortwaste/labels` symlink between `labels_4class` and `labels_8class`
to route ultralytics at the correct label set.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml
from ultralytics import YOLO
import ultralytics
import torch

VARIANT_NC = {"4class": 4, "8class": 8}


def read_best_metrics(model: YOLO, run_dir: Path) -> tuple[float | None, float | None]:
    try:
        metrics = getattr(model, "metrics", None)
        if metrics is not None and getattr(metrics, "box", None) is not None:
            return float(metrics.box.map), float(metrics.box.map50)
    except Exception:
        pass

    results_csv = run_dir / "results.csv"
    if not results_csv.exists():
        return None, None
    lines = results_csv.read_text().strip().splitlines()
    if len(lines) < 2:
        return None, None
    header = [c.strip() for c in lines[0].split(",")]
    try:
        i_map = header.index("metrics/mAP50-95(B)")
        i_map50 = header.index("metrics/mAP50(B)")
    except ValueError:
        return None, None
    last = [c.strip() for c in lines[-1].split(",")]
    try:
        return float(last[i_map]), float(last[i_map50])
    except (ValueError, IndexError):
        return None, None


def git_commit(project_root: Path) -> str | None:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if r.returncode == 0:
            return r.stdout.strip()
    except FileNotFoundError:
        pass
    return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--variant", choices=list(VARIANT_NC), required=True)
    p.add_argument("--name", type=str, required=True)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--optimizer", choices=["SGD", "AdamW", "Adam"], default="AdamW")
    p.add_argument("--lr0", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--device", type=str, default="mps")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    project_root = Path(__file__).resolve().parent.parent

    data_yaml_path = args.data.resolve()
    if not data_yaml_path.is_file():
        sys.exit(f"Data YAML not found: {data_yaml_path}")
    with data_yaml_path.open() as f:
        data_cfg = yaml.safe_load(f)
    nc = data_cfg.get("nc") or len(data_cfg.get("names", []))
    expected_nc = VARIANT_NC[args.variant]
    if nc != expected_nc:
        sys.exit(
            f"YAML nc mismatch: {data_yaml_path} has nc={nc}, "
            f"variant {args.variant!r} expects nc={expected_nc}"
        )

    project_dir = project_root / "runs" / "train"
    run_dir = project_dir / args.name
    if run_dir.exists():
        sys.exit(
            f"Run dir already exists: {run_dir}\n"
            f"Pick a new --name or delete the existing dir."
        )

    train_kwargs = dict(
        data=str(data_yaml_path),
        epochs=args.epochs,
        patience=args.patience,
        batch=8,
        imgsz=args.imgsz,
        optimizer=args.optimizer,
        lr0=args.lr0,
        cos_lr=True,
        seed=args.seed,
        device=args.device,
        project=str(project_dir),
        name=args.name,
        flipud=0.0,
        fliplr=0.5,
        exist_ok=False,
    )

    model = YOLO(args.model)
    model.train(**train_kwargs)

    best_map, best_map50 = read_best_metrics(model, run_dir)

    meta = {
        "cli_args": {
            k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
        },
        "train_overrides": train_kwargs,
        "variant": args.variant,
        "nc": nc,
        "data_yaml_resolved": str(data_yaml_path),
        "git_commit": git_commit(project_root),
        "ultralytics_version": ultralytics.__version__,
        "torch_version": torch.__version__,
        "best_pt": str(run_dir / "weights" / "best.pt"),
        "best_mAP50_95": best_map,
        "best_mAP50": best_map50,
    }
    meta_path = run_dir / "run_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"\nWrote {meta_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
