# AGENTS.md

Project context for Codex (or any LLM assistant) working on this repository.

## Project overview

This project reproduces the results of the paper **"SortWaste: A Densely Annotated Dataset for Object Detection in Industrial Waste Sorting"** (Inácio, Proença, Neves; arXiv:2601.02299, Jan 2026) using **YOLOv8** (Ultralytics) instead of the paper's YOLOv11. The paper benchmarks Faster R-CNN, TridentNet, RetinaNet, and YOLOv11 on the SortWaste dataset and introduces a **ClutterScore** metric for scene complexity. We replicate the YOLO benchmark on overall mAP only; the paper's ClutterScore-based robustness analysis (Figures 6, 7, Table 5) is out of scope.

**Reference targets (from paper Tables 6 and 7):**

- Full 8-class (YOLOv11): AP = 0.451, AP50 = 0.567
- Plastic-only 4-class (YOLOv11): AP = 0.597, AP50 = 0.753

YOLOv8 is expected to land within a few points of these — not identical. Track the gap; don't chase it.

## Dataset

**SortWaste** — 5,261 images, 87,252 bounding boxes, collected from a Portuguese MBT (Mechanical-Biological Treatment) facility. Top-down 1920×1080 frames of a waste conveyor belt at 5 fps.

**Splits (from paper Table 2):**

- Train: 3,705 images / 61,842 objects
- Val: 780 images / 13,065 objects
- Test: 776 images / 12,618 objects

Use the official split if released. If reconstructing: 200 consecutive frames per scene, drop the first 5 of each scene, then group scenes 70/15/15 keeping class distributions roughly even.

**Class layouts:**

8-class (full eval) — **order matches `configs/wastevision_8.yaml`**:
| ID | Class | Train | Val | Test |
|----|-------|-------|-----|------|
| 0 | PET | 11976 | 2108 | 2722 |
| 1 | HDPE | 16803 | 4972 | 3269 |
| 2 | Mixed_Soft_Plastic | 9077 | 1443 | 1817 |
| 3 | ECAL | 13649 | 2552 | 3026 |
| 4 | Metal | 945 | 277 | 215 |
| 5 | Cardboard | 1524 | 425 | 207 |
| 6 | Mixed_Rigid_Plastic | 7066 | 1120 | 1230 |
| 7 | PET_Oil | 802 | 168 | 132 |

4-class (plastic-only eval, paper Table 3) — **order matches `configs/wastevision_4.yaml`**:
| ID | Class | Total | Composition |
|----|-------|-------|-------------|
| 0 | PET | 17908 | PET + PET_Oil |
| 1 | HDPE | 25044 | HDPE |
| 2 | Mixed_Plastic | 21753 | Mixed_Soft + Mixed_Rigid |
| 3 | ECAL | 19227 | ECAL |

Cardboard and Metal are excluded from the 4-class run entirely.

Class imbalance is **intentional** — it reflects the real material distribution after upstream magnetic separation. Don't try to "fix" it by oversampling without flagging the change.

**Verified against local data (`notebooks/01_data_exploration.ipynb`, 8-class run):** per-split annotation totals match paper Table 2 *exactly* (61,842 / 13,065 / 12,618 = 87,525 total). No gap to caveat. 4-class totals also match Table 3 exactly. Class-ID order in the tables above is the order **the YAMLs use** and what training/eval will see — do not reorder without rebuilding labels.

## Repository layout

```
.
├── AGENTS.md                    # this file
├── ProjectTimeline.md           # running log of changes + decisions
├── README.md
├── configs/
│   ├── wastevision_8.yaml       # Ultralytics data config, 8 classes → yolo_8class/
│   └── wastevision_4.yaml       # Ultralytics data config, 4 classes → yolo_4class/
├── data/
│   └── sortwaste/
│       ├── sortwaste_raw/       # untouched upstream archive (COCO JSON + PNGs)
│       ├── labels_8class/{train,val,test}/*.txt    # canonical 8-class labels
│       ├── labels_4class/{train,val,test}/*.txt    # canonical 4-class labels
│       ├── yolo_8class/{train,val,test}/{images,labels}/   # what ultralytics actually reads
│       └── yolo_4class/{train,val,test}/{images,labels}/   # what ultralytics actually reads
├── src/
│   ├── convert_annotations.py   # verifier: COCO JSON ↔ released YOLO labels
│   ├── build_4class_labels.py   # verifier: 8-class → 4-class remap matches released set
│   ├── setup_yolo_trees.py      # build yolo_<variant>/ trees (hardlink images + copy labels)
│   ├── train.py                 # wraps ultralytics train
│   └── evaluate.py              # test-set eval, per-class table
├── runs/
│   ├── train/                   # ultralytics training output (gitignored)
│   └── eval/                    # ultralytics val output + per-run predictions.json
└── results/                     # generated tables, plots, CSVs
```

### Data layout — why two `yolo_<variant>/` trees

Ultralytics resolves image paths via `realpath` before deriving label paths
(`/images/ → /labels/` string substitution). A naive `data/sortwaste/images/`
symlinked to the raw image dir caused ultralytics to look for labels next to
the raw images — where ours don't live — so training silently ran on zero
labels. Two fixes work in principle: (a) reroute the YAML to a layout where
images and labels are real siblings; (b) hardlink images into a managed dir so
`realpath` returns the hardlink path itself. We use a hybrid: a single
`setup_yolo_trees.py` builds per-variant trees with **hardlinked** PNGs (no
disk cost) and **copied** label files (small, edit-safe). Both YAMLs point at
their respective `yolo_<variant>/` root.

Re-run `python src/setup_yolo_trees.py` whenever `labels_<variant>/` changes;
it is idempotent.

## Environment

- Python 3.10, conda env name `waste_vision`
- Key pins: `ultralytics==8.3.*`, PyTorch with **GPU support** (verify `torch.cuda.is_available()` returns `True` before training — pip-installed ultralytics may pull CPU-only torch on some systems)
- Other deps: opencv-python, pycocotools, pandas, numpy, matplotlib, seaborn, scikit-learn, tqdm

## Training recipe

Mirror the paper's regime to keep results comparable:

- Backbone: `yolov8m.pt` (closest capacity to yolov11m used by the paper)
- COCO-pretrained init
- Batch size: 8 (fixed, per paper)
- Image size: 1280 (paper doesn't specify). EDA confirms this is more than sufficient: 95–99% of boxes land in COCO's "large" bucket (side > 96 px) and zero are "small" (< 32 px), so there's no small-object recall pressure pushing toward higher resolution. Document this choice in any results.
- Augmentation: **disable vertical flip** (`flipud=0.0`). Frames are top-down conveyor views; class spatial priors are vertically asymmetric (per-class heatmaps cluster in a central band) and a vertical flip destroys that prior. Horizontal flip (`fliplr=0.5`, the default) is fine.
- Early stopping: patience 15, max 300 epochs
- Tune **only** learning rate and optimizer (paper does the same to keep comparisons fair). Sweep:
  - SGD lr=1e-2
  - AdamW lr=1e-3
  - AdamW lr=5e-4
- Select best on **validation** mAP50-95, then report **test** results
- Fix `seed=0` for reproducibility

**Density context** (from EDA, 8-class): median 17 objects/image, max 36, p95 ≈ 27. Batch=8 at imgsz=1280 means roughly ~140 boxes per batch on average — a moderately dense detection task, not extreme. Default YOLO anchor/loss settings are appropriate; no need to bump `box`/`cls` loss weights pre-emptively.

Re-run the LR/optimizer sweep for the 4-class experiment — the optimum may shift.

## Evaluation

Ultralytics reports `mAP50-95` (= paper's "AP") and `mAP50` (= paper's "AP50"). Per-class AP also available via `model.val()`.

## Scope note: ClutterScore

The paper's ClutterScore metric (§3.5) and its associated stratified analysis (Figures 6, 7, Table 5) are documented in the paper but not reproduced here. If revisiting this project to extend it, that's the natural next addition — the *O* term is already previewed in `notebooks/01_data_exploration.ipynb` section 8.

## Reproduction phases

1. **Data prep** — download/extract, convert annotations to YOLO format, build 8-class and 4-class label dirs, sanity-check object counts against Table 2/3
2. **Experiment A** — 8-class training, LR/optimizer sweep, test eval, build Table 6 analog
3. **Experiment B** — 4-class training, LR/optimizer sweep, test eval, build Table 7 analog

## Things to avoid

- **Don't change class definitions** without updating both labels and the data YAML. The 4-class remap (PET ⊇ PET_Oil, Mixed_Plastic = Mixed_Soft + Mixed_Rigid) must match Table 3 counts exactly.
- **Don't oversample minority classes** silently to boost rare-class AP. Class imbalance is a property of the benchmark.
- **Don't compare to paper numbers without flagging the YOLOv8 vs YOLOv11 substitution.** A 2–4 point AP gap is expected and not a bug.
- **Don't change image resolution mid-experiment.** Lock it for the full LR/optimizer sweep.
- **Don't trust ultralytics defaults silently** — log the resolved config (LR, optimizer, augmentations, imgsz) for every run.

## Known limitations of this reproduction

- YOLOv8 ≠ YOLOv11 architecturally; absolute numbers will differ.
- Per-class AP for rare classes (Cardboard ~207 test boxes, Metal ~215) has high variance. Single-digit swings are noise.
- Paper doesn't publish exact best hyperparameters or input resolution; some divergence is unavoidable.
- If reconstructing splits from raw video, frame indexing depends on extraction timing — splits won't match exactly even with the same scene logic.

## Useful commands

```bash
# Verify GPU / MPS
python -c "import torch; print('cuda', torch.cuda.is_available(), '/ mps', torch.backends.mps.is_available())"

# (One-time / after labels change) materialize yolo_<variant>/ trees
python src/setup_yolo_trees.py

# Verify released YOLO labels match the COCO source (one-time sanity check)
python src/convert_annotations.py \
    --coco-root data/sortwaste/sortwaste_raw/plastic_dataset/splited_dataset_plastic \
    --yolo-root data/sortwaste/sortwaste_raw/plastic_dataset/splited_dataset_plastic_yolo \
    --image-root data/sortwaste/sortwaste_raw/dataset/splited_all_dataset \
    --tolerance 1e-4

# Verify released 4-class labels are a pure remap of 8-class
python src/build_4class_labels.py \
    --src-labels data/sortwaste/labels_8class \
    --out-labels /tmp/labels_4class_derived \
    --released-4class data/sortwaste/labels_4class \
    --diff-report results/derived_vs_released_4class.csv

# Train 8-class (batch is locked at 8; flipud locked at 0)
python src/train.py --data configs/wastevision_8.yaml --model yolov8m.pt \
    --variant 8class --imgsz 1280 --epochs 300 --patience 15 \
    --optimizer AdamW --lr0 1e-3 --name yolov8m_8class_adamw_1e3 \
    --device mps --seed 0

# Evaluate on test split
python src/evaluate.py \
    --weights runs/train/yolov8m_8class_adamw_1e3/weights/best.pt \
    --data configs/wastevision_8.yaml --variant 8class \
    --imgsz 1280 --split test \
    --out results/eval_yolov8m_8class.csv \
    --device mps
```

## Paper reference

Inácio, S., Proença, H., Neves, J. C. _SortWaste: A Densely Annotated Dataset for Object Detection in Industrial Waste Sorting._ arXiv:2601.02299, January 2026.

Key sections to revisit when in doubt:

- §3.2 — class definitions (verbatim material descriptions)
- §3.3 — split construction (200-frame scenes, drop first 5)
- §4.1 — training protocol
- Tables 2, 3, 6, 7 — target counts and reference numbers
