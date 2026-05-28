# Results — YOLOv8m on SortWaste (4-class plastic)

YOLOv8m reproduces YOLOv11's reported plastic-detection performance on SortWaste
within run-to-run noise — overall AP50 0.753 vs 0.753, AP 0.598 vs 0.597 —
using AdamW at lr=5e-4.

## Setup

Ran `yolov8m` initialized from COCO-pretrained weights with batch size 8,
imgsz 1280, cosine LR schedule, early-stopping patience 15, max 300 epochs,
a single seed (0), and vertical flip disabled (`flipud=0`, `fliplr=0.5`). Only
the 4-class plastic split was evaluated (PET, HDPE, Mixed_Plastic, ECAL). The
reference target is the YOLOv11 column of paper Table 7.

## The sweep

**Table A — hyperparameter sweep** (sorted by val_mAP50_95 desc)

| Run | Optimizer | LR | Val AP50-95 | Test AP50 | Test AP |
|---|---|---|---|---|---|
| yolov8m_4class_adamw_5e-04 ✓ | AdamW | 5e-04 | 0.639 | 0.753 | 0.598 |
| yolov8m_4class_adamw_1e-03 | AdamW | 1e-03 | 0.626 | 0.727 | 0.571 |
| yolov8m_4class_sgd_1e-02 | SGD | 1e-02 | 0.619 | 0.735 | 0.587 |

Three optimizer/LR configurations were tried; all three cluster within about
two points on val AP50-95 (0.619–0.639). AdamW at lr=5e-4 edged ahead and was
selected by the validation rule. LR sensitivity is visible: dropping AdamW from
5e-4 to 1e-3 cost roughly 2.6 test-AP50 points (0.753 → 0.727) on the same
optimizer.

## Winner vs paper

**Table B — winner vs paper, per class**

| Class | YOLOv8m (ours) AP50 | YOLOv11 (paper) AP50 | Δ |
|---|---|---|---|
| PET | 0.859 | 0.872 | -0.013 |
| HDPE | 0.720 | 0.729 | -0.009 |
| Mixed_Plastic | 0.628 | 0.624 | +0.004 |
| ECAL | 0.803 | 0.786 | +0.017 |
| Overall AP50 | 0.753 | 0.753 | -0.000 |
| Overall AP(50-95) | 0.598 | 0.597 | +0.001 |

Overall numbers land on the paper's to within a point on both AP and AP50. The
per-class AP50 ordering (PET > ECAL > HDPE > Mixed_Plastic) matches the paper
exactly. Mixed_Plastic is the hardest class in both runs, consistent with the
paper's own observation (§4.3) that merging rigid and soft plastics into a
single class produces a heterogeneous category that is harder to detect.

## Caveats

- Single seed. YOLO has roughly one point of run-to-run variance; the exact
  0.753 AP50 match is partly fortunate. The defensible claim is
  "indistinguishable from YOLOv11 within noise," not "exactly reproduced as a
  method."
- YOLOv8m ≠ YOLOv11 architecturally. Convergence to the same numbers is the
  finding here, not an assumption going in.
- 4-class plastic scope only. The paper's 8-class benchmark (Table 6) was not
  reproduced.
- The paper's ClutterScore robustness analysis (Figures 6/7, Table 5) was out
  of scope; only overall and per-class mAP are reported.

## Reproduce

```bash
# Train the winning config
python src/train.py \
    --data configs/wastevision_4.yaml --model yolov8m.pt \
    --variant 4class --imgsz 1280 --epochs 300 --patience 15 \
    --optimizer AdamW --lr0 5e-4 \
    --name yolov8m_4class_adamw_5e-04 \
    --device 0 --seed 0

# Evaluate on the test split
python src/evaluate.py \
    --weights runs/train/yolov8m_4class_adamw_5e-04/weights/best.pt \
    --data configs/wastevision_4.yaml --variant 4class \
    --imgsz 1280 --split test \
    --out results/eval_yolov8m_4class_adamw_5e-04.csv \
    --device 0
```
