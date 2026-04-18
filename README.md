# HW2: Digit Detection with DETR

**Name:** 陳沛妤 (Pei-Yu Chen)  
**Student ID:** 314560017  
**Course:** Visual Recognition using Deep Learning (2026 Spring)

## Introduction

This project implements a digit detection system using **DETR (DEtection TRansformer)** with a **ResNet-50** backbone. The model is fine-tuned on a custom digit dataset (digits 0–9) to predict bounding boxes and class labels for each digit in an image. The output follows the COCO format for evaluation on CodaBench.

## Environment Setup

### Prerequisites
- Python 3.9 or higher
- CUDA-compatible GPU (tested on NVIDIA RTX 5070)

### Installation

```bash
# Create and activate virtual environment
python -m venv venv

# Windows
.\venv\Scripts\activate

# Linux / macOS
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
- PyTorch >= 2.0.0
- Torchvision >= 0.15.0
- Transformers >= 4.30.0
- pycocotools >= 2.0.7
- NumPy, Pillow, Matplotlib, scikit-learn, tqdm, SciPy

## Usage

### Training

```bash
python train.py \
    --data_dir ./nycu-hw2-data \
    --output_dir ./output \
    --epochs 50 \
    --batch_size 8 \
    --lr 1e-4 \
    --lr_backbone 1e-5 \
    --image_size 480 \
    --num_workers 0 \
    --eval_interval 5
```

Key arguments:
| Argument | Default | Description |
|---|---|---|
| `--data_dir` | `./nycu-hw2-data` | Path to dataset root |
| `--epochs` | `50` | Number of training epochs |
| `--batch_size` | `8` | Training batch size |
| `--lr` | `1e-4` | Learning rate for transformer |
| `--lr_backbone` | `1e-5` | Learning rate for ResNet-50 backbone |
| `--image_size` | `480` | Input image resize dimension |
| `--eval_interval` | `5` | mAP evaluation frequency (epochs) |

### Inference

```bash
python inference.py \
    --data_dir ./nycu-hw2-data \
    --checkpoint ./output/best_map_model.pth \
    --image_size 480 \
    --score_threshold 0.1
```

This generates `pred.json` and a submission zip file in the `./output` directory.

## Project Structure

```
.
├── train.py          # Training pipeline with mixed-precision support
├── inference.py      # Inference and submission generation
├── model.py          # DETR model builder (ResNet-50 backbone)
├── dataset.py        # Dataset classes and data augmentation
├── utils.py          # Utility functions (EarlyStopping, mAP evaluation)
├── requirements.txt  # Python dependencies
└── output/           # Checkpoints, predictions, training curves
```

## Performance Snapshot

| Metric | Value |
|---|---|
| **Public Leaderboard mAP** | **0.35** |
| Best Validation mAP | 0.4327 |
| Best Validation Loss | 1.0004 |
| Training Epochs | 50 |
| Total Training Time | ~916 minutes (~15.3 hours) |

### Training Curves

The training loss steadily decreases across all 50 epochs while the validation loss stabilizes around 1.0, indicating healthy convergence without significant overfitting.

<img width="2670" height="718" alt="image" src="https://github.com/user-attachments/assets/e31f3e5a-d9ef-4844-9dbf-4feb84337252" />

