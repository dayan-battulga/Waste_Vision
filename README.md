# Waste_Vision — YOLOv8 Plastic Model Detection
A YoloV8 Model fine-tuned for plastic detection and classification under a conveyer belt of foodwaste being wheeled in real-time. The mission is to eliminate plastic contamination and redirect compostable waste into composting facilities. 

## Headline result

| Metric            | YOLOv8m (this work) |
|-------------------|---------------------|
| Overall AP50      | 0.753               |
| Overall AP(50-95) | 0.598               |

See [results/RESULTS.md](results/RESULTS.md) for the per-class table,
the full LR/optimizer sweep, and caveats.

## Performance figures

All figures below are from the winning run
(`yolov8m_4class_adamw_5e-04`). Source artifacts live under
`runs/train/.../` and `runs/eval/train_test/` (gitignored — regenerable via
`src/train.py` + `src/evaluate.py`); the copies in [docs/figures/](docs/figures/)
are tracked so they render here.

## Dataset

SortWaste was used as the dataset to train the model, as it is the most identical and similar to the conditions Waste_Vision will be deployed on in the real-world. There is also explicit plastic labels (PET, HDPE, etc.) to classify different types of plastic, which is the main task of the model. The dataset was downloaded locally and uploaded to Google Drive to run the experiments on Colab.

**Training curves (val mAP + losses across epochs)**

![training curves](docs/figures/training_curves.png)

**Test-split precision-recall** (left) and **normalized confusion matrix**
(right). PR curve overall mAP@0.5 = 0.753; PET is the easiest class, Mixed_Plastic is the hardest.

| | |
|---|---|
| ![PR curve](docs/figures/test_pr_curve.png) | ![confusion matrix](docs/figures/test_confusion_matrix.png) |

**Sample test-batch detections.** Ground-truth labels (left) and YOLOv8m
predictions (right) on the same test batch.

| Ground truth | Predictions |
|---|---|
| ![labels](docs/figures/test_batch_labels.jpg) | ![predictions](docs/figures/test_batch_predictions.jpg) |

## What's in the repo

- [src/](src/) — training, eval, label verifiers, webcam demo
- [configs/](configs/) — Ultralytics data YAMLs (4-class and 8-class)
- [notebooks/](notebooks/) — EDA (`01_data_exploration.ipynb`) and Colab training driver (`02_colab_training.ipynb`)
- [results/](results/) — eval CSVs, comparison tables, [RESULTS.md](results/RESULTS.md)
- [ProjectTimeline.md](ProjectTimeline.md) — running decision log

## Reproducing this

### Environment

```bash
conda create -n waste_vision python=3.10 -y
conda activate waste_vision
pip install -r requirements.txt
```

Verify GPU/MPS is wired up before training:

```bash
python -c "import torch; print('cuda', torch.cuda.is_available(), '/ mps', torch.backends.mps.is_available())"
```

### Training (Colab recommended)

GPU training of the full 300-epoch run is most practical on Colab — see
[notebooks/02_colab_training.ipynb](notebooks/02_colab_training.ipynb) for the
driver notebook. Local training works via `src/train.py` but is slow on CPU/MPS.

### Eval

Reproduce the winning run's test-set numbers:

```bash
python src/evaluate.py \
    --weights runs/train/yolov8m_4class_adamw_5e-04/train/weights/best.pt \
    --data configs/wastevision_4.yaml \
    --variant 4class --imgsz 1280 --split test \
    --out results/eval_yolov8m_4class_adamw_5e-04.csv \
    --device mps
```

## Live demo (optional)

[src/webcam_infer.py](src/webcam_infer.py) runs the trained detector on a live
webcam feed for visual demonstration. The model was trained on top-down
conveyor frames under uniform facility lighting, so handheld webcam input is
out of distribution — the demo illustrates the failure mode about as much as
the success mode.

## Citation

> Inácio, S., Proença, H., Neves, J. C. *SortWaste: A Densely Annotated
> Dataset for Object Detection in Industrial Waste Sorting.* arXiv:2601.02299,
> January 2026.
