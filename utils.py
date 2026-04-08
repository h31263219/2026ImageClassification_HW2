"""Utility functions for digit detection training.

Includes plotting, early stopping, seed setting, and mAP evaluation.
"""

import os
import random
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EarlyStopping:
    """Early stopping to terminate training when metric stops improving.

    Args:
        patience: Number of epochs to wait before stopping.
        mode: 'min' for loss, 'max' for metrics like mAP.
        min_delta: Minimum change to qualify as an improvement.
    """

    def __init__(
        self,
        patience: int = 10,
        mode: str = 'max',
        min_delta: float = 0.0,
    ) -> None:
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_score: Optional[float] = None
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        """Check if training should stop.

        Args:
            score: Current metric value.

        Returns:
            True if training should stop.
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True

        return False


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: str = 'training_curves.png',
) -> None:
    """Plot training and validation loss/mAP curves.

    Args:
        history: Dictionary with training history.
        save_path: Path to save the plot.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss curve
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # mAP curve
    if 'val_map' in history:
        axes[1].plot(history['val_map'], label='Val mAP', color='green')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('mAP')
        axes[1].set_title('Validation mAP')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    # Learning rate curve
    if 'lr' in history:
        axes[2].plot(history['lr'], label='Learning Rate', color='orange')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('LR')
        axes[2].set_title('Learning Rate Schedule')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to {save_path}")


def convert_to_xywh(boxes: torch.Tensor) -> torch.Tensor:
    """Convert [cx, cy, w, h] to [x_min, y_min, w, h].

    Args:
        boxes: Tensor of shape (N, 4) in [cx, cy, w, h] format.

    Returns:
        Tensor of shape (N, 4) in [x_min, y_min, w, h] format.
    """
    x_min = boxes[:, 0] - boxes[:, 2] / 2
    y_min = boxes[:, 1] - boxes[:, 3] / 2
    w = boxes[:, 2]
    h = boxes[:, 3]
    return torch.stack([x_min, y_min, w, h], dim=1)


def convert_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert [cx, cy, w, h] to [x_min, y_min, x_max, y_max].

    Args:
        boxes: Tensor of shape (N, 4) in [cx, cy, w, h] format.

    Returns:
        Tensor of shape (N, 4) in [x_min, y_min, x_max, y_max] format.
    """
    x_min = boxes[:, 0] - boxes[:, 2] / 2
    y_min = boxes[:, 1] - boxes[:, 3] / 2
    x_max = boxes[:, 0] + boxes[:, 2] / 2
    y_max = boxes[:, 1] + boxes[:, 3] / 2
    return torch.stack([x_min, y_min, x_max, y_max], dim=1)
