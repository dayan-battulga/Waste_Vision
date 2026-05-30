"""Live webcam demo for the 4-class SortWaste plastic detector.

DEMO TOOL. The checkpoint was trained on top-down conveyor-belt frames under
uniform facility lighting. Webcam input (oblique angle, room lighting, single
handheld items) is out of distribution — expect misses and noisy boxes.

Opens a webcam, runs YOLOv8 detection per frame, and displays an annotated
window with rolling FPS, current inference time, per-class counts, and the
active confidence threshold. Keybindings let you snapshot, pause, and adjust
the threshold live.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter, deque
from datetime import datetime
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

DEFAULT_WEIGHTS = (
    PROJECT_ROOT
    / "runs"
    / "train"
    / "yolov8m_4class_adamw_5e-04"
    / "train"
    / "weights"
    / "best.pt"
)
DEFAULT_SNAPSHOT_DIR = PROJECT_ROOT / "results" / "webcam_snapshots"

WARNING_TEXT = (
    "WARNING: demo only. Model trained on top-down conveyor frames at facility "
    "lighting; webcam input is out of distribution, so misses and noisy boxes "
    "are expected."
)

KEYBIND_HELP = (
    "Controls:\n"
    "  q / ESC   quit\n"
    "  s         save annotated snapshot\n"
    "  + / -     raise / lower confidence threshold by 0.05 (clamped [0.05, 0.95])\n"
    "  space     pause / resume"
)

WINDOW_TITLE = "SortWaste - live plastic detection"


def resolve_device(requested: str) -> str:
    req = requested.lower()
    if req == "mps" and not torch.backends.mps.is_available():
        print("WARNING: --device mps requested but MPS is not available; falling back to cpu.")
        return "cpu"
    if req == "cuda" and not torch.cuda.is_available():
        print("WARNING: --device cuda requested but CUDA is not available; falling back to cpu.")
        return "cpu"
    return requested


def open_camera(camera_index: int) -> cv2.VideoCapture | None:
    if sys.platform == "darwin":
        cap = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
    else:
        cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(
            f"ERROR: Could not open camera index {camera_index}. "
            "On macOS check System Settings > Privacy & Security > Camera "
            "and grant access to your terminal / IDE.",
            file=sys.stderr,
        )
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap


def draw_overlay(
    frame,
    *,
    names: dict[int, str],
    fps: float,
    infer_ms: float,
    counts: dict[int, int],
    conf: float,
    paused: bool,
):
    box_w, box_h = 560, 60
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (box_w, box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    line1 = (
        f"FPS: {fps:5.1f}   infer: {infer_ms:5.1f} ms   conf>={conf:.2f}"
        f"{'  [PAUSED]' if paused else ''}"
    )
    line2 = "  ".join(f"{names[i]}: {counts.get(i, 0)}" for i in sorted(names))

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, line1, (10, 22), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, line2, (10, 48), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def save_snapshot(frame, snapshot_dir: Path) -> Path:
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    path = snapshot_dir / f"snapshot_{datetime.now():%Y%m%d_%H%M%S_%f}.png"
    cv2.imwrite(str(path), frame)
    return path


def count_classes(result, names: dict[int, str]) -> dict[int, int]:
    counts: Counter = Counter()
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.cls is None:
        return {i: 0 for i in names}
    for c in boxes.cls.tolist():
        counts[int(c)] += 1
    return {i: counts.get(i, 0) for i in names}


def normalize_names(model) -> dict[int, str]:
    raw = getattr(model, "names", None)
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return {int(k): str(v) for k, v in raw.items()}
    return {i: str(n) for i, n in enumerate(raw)}


def run_test_image(
    model,
    *,
    names: dict[int, str],
    image_path: Path,
    imgsz: int,
    conf: float,
    device: str,
    snapshot_dir: Path,
) -> int:
    if not image_path.is_file():
        sys.exit(f"Test image not found: {image_path}")
    frame = cv2.imread(str(image_path))
    if frame is None:
        sys.exit(f"Could not read image (cv2.imread returned None): {image_path}")

    t0 = time.perf_counter()
    results = model.predict(
        frame, imgsz=imgsz, conf=conf, device=device, verbose=False
    )
    infer_ms = (time.perf_counter() - t0) * 1000.0

    result = results[0]
    annotated = result.plot()
    counts = count_classes(result, names)
    annotated = draw_overlay(
        annotated,
        names=names,
        fps=0.0,
        infer_ms=infer_ms,
        counts=counts,
        conf=conf,
        paused=False,
    )

    saved_path = save_snapshot(annotated, snapshot_dir)
    counts_str = " ".join(f"{names[i]}={counts.get(i, 0)}" for i in sorted(names))
    print(f"Test image: {image_path}")
    print(f"Inference time: {infer_ms:.1f} ms")
    print(f"Per-class counts: {counts_str}")
    print(f"Saved annotated to: {saved_path}")
    return 0


def run_live(
    model,
    *,
    names: dict[int, str],
    camera_index: int,
    imgsz: int,
    conf: float,
    device: str,
    snapshot_dir: Path,
) -> int:
    cap = open_camera(camera_index)
    if cap is None:
        return 2

    frame_times: deque[float] = deque(maxlen=30)
    frames_processed = 0
    snapshots_saved = 0
    paused = False
    last_clean_annotated = None
    last_counts: dict[int, int] = {i: 0 for i in names}
    display_frame = None

    try:
        while True:
            if not paused:
                ok, frame = cap.read()
                if not ok or frame is None:
                    print("Camera read failed; exiting.", file=sys.stderr)
                    break

                t0 = time.perf_counter()
                results = model.predict(
                    frame, imgsz=imgsz, conf=conf, device=device, verbose=False
                )
                t1 = time.perf_counter()
                infer_ms = (t1 - t0) * 1000.0

                frame_times.append(t1)
                if len(frame_times) > 1:
                    span = frame_times[-1] - frame_times[0]
                    rolling_fps = (len(frame_times) - 1) / span if span > 0 else 0.0
                else:
                    rolling_fps = 0.0

                result = results[0]
                clean_annotated = result.plot()
                last_clean_annotated = clean_annotated
                last_counts = count_classes(result, names)
                display_frame = draw_overlay(
                    clean_annotated.copy(),
                    names=names,
                    fps=rolling_fps,
                    infer_ms=infer_ms,
                    counts=last_counts,
                    conf=conf,
                    paused=False,
                )
                frames_processed += 1
            elif last_clean_annotated is not None:
                display_frame = draw_overlay(
                    last_clean_annotated.copy(),
                    names=names,
                    fps=0.0,
                    infer_ms=0.0,
                    counts=last_counts,
                    conf=conf,
                    paused=True,
                )

            if display_frame is not None:
                cv2.imshow(WINDOW_TITLE, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            elif key == ord("s") and display_frame is not None:
                path = save_snapshot(display_frame, snapshot_dir)
                snapshots_saved += 1
                print(f"Saved snapshot: {path}")
            elif key in (ord("+"), ord("=")):
                conf = min(round(conf + 0.05, 2), 0.95)
                print(f"Confidence threshold -> {conf:.2f}")
            elif key == ord("-"):
                conf = max(round(conf - 0.05, 2), 0.05)
                print(f"Confidence threshold -> {conf:.2f}")
            elif key == ord(" "):
                paused = not paused
                print(f"{'Paused' if paused else 'Resumed'}.")
                if not paused:
                    frame_times.clear()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if len(frame_times) > 1:
        span = frame_times[-1] - frame_times[0]
        avg_fps = (len(frame_times) - 1) / span if span > 0 else 0.0
    else:
        avg_fps = 0.0
    print(
        f"Session summary: frames processed: {frames_processed} | "
        f"average FPS (last window): {avg_fps:.1f} | snapshots saved: {snapshots_saved}"
    )
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.4)
    p.add_argument("--device", type=str, default="mps")
    p.add_argument("--snapshot-dir", type=Path, default=DEFAULT_SNAPSHOT_DIR)
    p.add_argument("--test-image", type=Path, default=None,
                   help=argparse.SUPPRESS)
    args = p.parse_args()

    print(WARNING_TEXT)

    if not args.weights.is_file():
        sys.exit(f"Weights not found: {args.weights}")
    if not (0.05 <= args.conf <= 0.95):
        sys.exit(f"--conf must be in [0.05, 0.95]; got {args.conf}")

    device = resolve_device(args.device)
    model = YOLO(str(args.weights))
    names = normalize_names(model)
    if not names:
        sys.exit("Model has no class names; cannot run.")

    names_line = ", ".join(f"{i}={names[i]}" for i in sorted(names))
    print(f"Loaded model from {args.weights}")
    print(f"Device: {device}")
    print(f"Classes: {names_line}")

    if args.test_image is not None:
        return run_test_image(
            model,
            names=names,
            image_path=args.test_image,
            imgsz=args.imgsz,
            conf=args.conf,
            device=device,
            snapshot_dir=args.snapshot_dir,
        )

    print(KEYBIND_HELP)
    return run_live(
        model,
        names=names,
        camera_index=args.camera,
        imgsz=args.imgsz,
        conf=args.conf,
        device=device,
        snapshot_dir=args.snapshot_dir,
    )


if __name__ == "__main__":
    sys.exit(main())
