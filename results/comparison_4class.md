# 4-class comparison: YOLOv8m vs paper YOLOv11

**Table A — hyperparameter sweep** (sorted by val_mAP50_95 desc)

| Run | Optimizer | LR | Val AP50-95 | Test AP50 | Test AP |
|---|---|---|---|---|---|
| yolov8m_4class_adamw_5e-04 ✓ | AdamW | 5e-04 | 0.639 | 0.753 | 0.598 |
| yolov8m_4class_adamw_1e-03 | AdamW | 1e-03 | 0.626 | 0.727 | 0.571 |
| yolov8m_4class_sgd_1e-02 | SGD | 1e-02 | 0.619 | 0.735 | 0.587 |

**Table B — winner vs paper, per class**

| Class | YOLOv8m (ours) AP50 | YOLOv11 (paper) AP50 | Δ |
|---|---|---|---|
| PET | 0.859 | 0.872 | -0.013 |
| HDPE | 0.720 | 0.729 | -0.009 |
| Mixed_Plastic | 0.628 | 0.624 | +0.004 |
| ECAL | 0.803 | 0.786 | +0.017 |
| Overall AP50 | 0.753 | 0.753 | -0.000 |
| Overall AP(50-95) | 0.598 | 0.597 | +0.001 |
