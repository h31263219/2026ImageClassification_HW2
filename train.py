"""Training script for digit detection using DETR.

Fine-tunes a pretrained DETR model with ResNet-50 backbone
on a digit detection dataset in COCO format.

Usage:
    python train.py --data_dir ./nycu-hw2-data --epochs 50
"""

import argparse
import json
import os
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DetrImageProcessor

from dataset import (
    CocoDetectionDataset,
    DetectionTransform,
    collate_fn,
)
from model import build_model, get_model_info
from utils import EarlyStopping, plot_training_curves, set_seed


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description='Train DETR for digit detection',
    )
    parser.add_argument(
        '--data_dir', type=str,
        default='./nycu-hw2-data',
        help='Path to dataset root directory',
    )
    parser.add_argument(
        '--output_dir', type=str, default='./output',
        help='Directory to save outputs',
    )
    parser.add_argument(
        '--epochs', type=int, default=100,
        help='Number of training epochs',
    )
    parser.add_argument(
        '--batch_size', type=int, default=8,
        help='Training batch size',
    )
    parser.add_argument(
        '--lr', type=float, default=1e-4,
        help='Initial learning rate',
    )
    parser.add_argument(
        '--lr_backbone', type=float, default=1e-5,
        help='Learning rate for backbone',
    )
    parser.add_argument(
        '--weight_decay', type=float, default=1e-4,
        help='Weight decay for optimizer',
    )
    parser.add_argument(
        '--num_workers', type=int, default=4,
        help='Number of data loading workers',
    )
    parser.add_argument(
        '--patience', type=int, default=15,
        help='Early stopping patience',
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed',
    )
    parser.add_argument(
        '--num_queries', type=int, default=50,
        help='Number of DETR object queries',
    )
    parser.add_argument(
        '--image_size', type=int, default=800,
        help='Resize shortest side to this value',
    )
    parser.add_argument(
        '--eval_interval', type=int, default=1,
        help='Evaluate every N epochs',
    )
    parser.add_argument(
        '--resume', type=str, default=None,
        help='Path to checkpoint to resume from',
    )
    return parser.parse_args()


def prepare_detr_inputs(
    images: List[torch.Tensor],
    targets: List[Dict],
    processor: DetrImageProcessor,
    device: torch.device,
) -> Tuple[Dict, List[Dict]]:
    """Prepare inputs for DETR model using the image processor.

    Args:
        images: List of image tensors.
        targets: List of target dictionaries.
        processor: DETR image processor.
        device: Target device.

    Returns:
        Tuple of (model_inputs, formatted_targets).
    """
    # Pad images to same size within batch
    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)

    batch_size = len(images)
    padded = torch.zeros(batch_size, 3, max_h, max_w)
    pixel_mask = torch.zeros(batch_size, max_h, max_w, dtype=torch.long)

    for i, img in enumerate(images):
        _, h, w = img.shape
        padded[i, :, :h, :w] = img
        pixel_mask[i, :h, :w] = 1

    # Format targets for DETR loss computation
    formatted_targets = []
    for t in targets:
        formatted_targets.append({
            'class_labels': t['labels'].to(device),
            'boxes': t['boxes'].to(device),
        })

    inputs = {
        'pixel_values': padded.to(device),
        'pixel_mask': pixel_mask.to(device),
    }

    return inputs, formatted_targets


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    processor: DetrImageProcessor,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    """Train the model for one epoch.

    Args:
        model: DETR model.
        dataloader: Training data loader.
        processor: DETR image processor.
        optimizer: Optimizer.
        scaler: Gradient scaler for mixed precision.
        device: Target device.

    Returns:
        Tuple of (average_loss, loss_components_dict).
    """
    model.train()
    running_loss = 0.0
    running_loss_ce = 0.0
    running_loss_bbox = 0.0
    running_loss_giou = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc='Training', leave=False)
    for images, targets in pbar:
        inputs, formatted_targets = prepare_detr_inputs(
            images, targets, processor, device,
        )

        optimizer.zero_grad(set_to_none=True)

        with autocast('cuda'):
            outputs = model(
                pixel_values=inputs['pixel_values'],
                pixel_mask=inputs['pixel_mask'],
                labels=formatted_targets,
            )
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        loss_dict = outputs.loss_dict
        running_loss_ce += loss_dict.get(
            'loss_ce', torch.tensor(0.0),
        ).item()
        running_loss_bbox += loss_dict.get(
            'loss_bbox', torch.tensor(0.0),
        ).item()
        running_loss_giou += loss_dict.get(
            'loss_giou', torch.tensor(0.0),
        ).item()
        num_batches += 1

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'ce': f'{loss_dict.get("loss_ce", 0):.3f}',
            'bbox': f'{loss_dict.get("loss_bbox", 0):.3f}',
        })

    avg_loss = running_loss / max(num_batches, 1)
    loss_components = {
        'loss_ce': running_loss_ce / max(num_batches, 1),
        'loss_bbox': running_loss_bbox / max(num_batches, 1),
        'loss_giou': running_loss_giou / max(num_batches, 1),
    }
    return avg_loss, loss_components


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    processor: DetrImageProcessor,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    """Validate the model.

    Args:
        model: DETR model.
        dataloader: Validation data loader.
        processor: DETR image processor.
        device: Target device.

    Returns:
        Tuple of (average_loss, loss_components_dict).
    """
    model.eval()
    running_loss = 0.0
    running_loss_ce = 0.0
    running_loss_bbox = 0.0
    running_loss_giou = 0.0
    num_batches = 0

    for images, targets in tqdm(dataloader, desc='Validating', leave=False):
        inputs, formatted_targets = prepare_detr_inputs(
            images, targets, processor, device,
        )

        with autocast('cuda'):
            outputs = model(
                pixel_values=inputs['pixel_values'],
                pixel_mask=inputs['pixel_mask'],
                labels=formatted_targets,
            )

        running_loss += outputs.loss.item()
        loss_dict = outputs.loss_dict
        running_loss_ce += loss_dict.get(
            'loss_ce', torch.tensor(0.0),
        ).item()
        running_loss_bbox += loss_dict.get(
            'loss_bbox', torch.tensor(0.0),
        ).item()
        running_loss_giou += loss_dict.get(
            'loss_giou', torch.tensor(0.0),
        ).item()
        num_batches += 1

    avg_loss = running_loss / max(num_batches, 1)
    loss_components = {
        'loss_ce': running_loss_ce / max(num_batches, 1),
        'loss_bbox': running_loss_bbox / max(num_batches, 1),
        'loss_giou': running_loss_giou / max(num_batches, 1),
    }
    return avg_loss, loss_components


@torch.no_grad()
def evaluate_map(
    model: nn.Module,
    dataloader: DataLoader,
    processor: DetrImageProcessor,
    device: torch.device,
    score_threshold: float = 0.5,
) -> float:
    """Evaluate mAP on the validation set using pycocotools.

    Args:
        model: DETR model.
        dataloader: Validation data loader.
        processor: DETR image processor.
        device: Target device.
        score_threshold: Minimum score to consider a detection.

    Returns:
        mAP value.
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    import json
    import tempfile

    model.eval()
    all_predictions = []

    # Collect ground truth info
    dataset = dataloader.dataset

    for images, targets in tqdm(
        dataloader, desc='Evaluating mAP', leave=False,
    ):
        inputs, _ = prepare_detr_inputs(
            images, targets, processor, device,
        )

        with autocast('cuda'):
            outputs = model(
                pixel_values=inputs['pixel_values'],
                pixel_mask=inputs['pixel_mask'],
            )

        # Post-process predictions
        # outputs.logits: (batch, num_queries, num_classes + 1)
        # outputs.pred_boxes: (batch, num_queries, 4) in [cx, cy, w, h]
        probs = outputs.logits.softmax(-1)
        # Last class is "no object" for DETR
        scores, labels = probs[..., :-1].max(-1)

        for i in range(len(images)):
            img_id = targets[i]['image_id'].item()
            orig_h, orig_w = targets[i]['orig_size'].tolist()

            # Filter by score threshold
            keep = scores[i] > score_threshold
            pred_scores = scores[i][keep]
            pred_labels = labels[i][keep]
            pred_boxes = outputs.pred_boxes[i][keep]

            # Convert boxes from normalized [cx, cy, w, h] to
            # absolute [x_min, y_min, w, h]
            if len(pred_boxes) > 0:
                pred_boxes = pred_boxes.cpu()
                cx = pred_boxes[:, 0] * orig_w
                cy = pred_boxes[:, 1] * orig_h
                bw = pred_boxes[:, 2] * orig_w
                bh = pred_boxes[:, 3] * orig_h
                x_min = cx - bw / 2
                y_min = cy - bh / 2

                for j in range(len(pred_scores)):
                    # Map 0-indexed label back to category_id (1-indexed)
                    cat_id = pred_labels[j].item() + 1
                    all_predictions.append({
                        'image_id': img_id,
                        'category_id': cat_id,
                        'bbox': [
                            x_min[j].item(),
                            y_min[j].item(),
                            bw[j].item(),
                            bh[j].item(),
                        ],
                        'score': pred_scores[j].item(),
                    })

    if not all_predictions:
        return 0.0

    # Use pycocotools for mAP evaluation
    ann_file = os.path.join(
        dataloader.dataset.img_dir, '..', 'valid.json',
    )
    ann_file = os.path.normpath(ann_file)

    try:
        coco_gt = COCO(ann_file)

        # Write predictions to temp file
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False,
        ) as f:
            json.dump(all_predictions, f)
            pred_file = f.name

        coco_dt = coco_gt.loadRes(pred_file)
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        os.unlink(pred_file)
        return coco_eval.stats[0]  # mAP@[0.5:0.95]
    except Exception as e:
        print(f"Warning: mAP evaluation failed: {e}")
        return 0.0


def main() -> None:
    """Main training loop."""
    args = parse_args()
    set_seed(args.seed)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Data paths
    train_img_dir = os.path.join(args.data_dir, 'train')
    valid_img_dir = os.path.join(args.data_dir, 'valid')
    train_ann = os.path.join(args.data_dir, 'train.json')
    valid_ann = os.path.join(args.data_dir, 'valid.json')

    # Transforms
    train_transform = DetectionTransform(
        size=args.image_size, train=True,
    )
    val_transform = DetectionTransform(
        size=args.image_size, train=False,
    )

    # Datasets
    train_dataset = CocoDetectionDataset(
        img_dir=train_img_dir,
        ann_file=train_ann,
        transforms=train_transform,
    )
    val_dataset = CocoDetectionDataset(
        img_dir=valid_img_dir,
        ann_file=valid_ann,
        transforms=val_transform,
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Number of classes: {train_dataset.num_classes}")

    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if args.num_workers > 0 else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
    )

    # Model
    model = build_model(
        num_classes=train_dataset.num_classes,
        pretrained=True,
        num_queries=args.num_queries,
    )
    model = model.to(device)

    # Print model info
    info = get_model_info(model)
    print(f"Total parameters: {info['total_params']:,}")
    print(f"Trainable parameters: {info['trainable_params']:,}")
    print(f"Backbone parameters: {info['backbone_params']:,}")

    # DETR Image Processor (for reference, not used directly)
    processor = DetrImageProcessor.from_pretrained(
        'facebook/detr-resnet-50',
    )

    # Optimizer: Different LR for backbone and rest
    backbone_params = []
    other_params = []
    for name, param in model.named_parameters():
        if 'backbone' in name:
            backbone_params.append(param)
        else:
            other_params.append(param)

    optimizer = AdamW([
        {'params': backbone_params, 'lr': args.lr_backbone},
        {'params': other_params, 'lr': args.lr},
    ], weight_decay=args.weight_decay)

    # Scheduler
    scheduler = CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6,
    )

    # Mixed precision
    scaler = GradScaler()

    # Early stopping (monitor validation loss)
    early_stopping = EarlyStopping(patience=args.patience, mode='min')

    # Resume from checkpoint
    start_epoch = 1
    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        # Skip optimizer state to avoid potential memory issues
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {checkpoint['epoch']} (model weights only, fresh optimizer)")

    # Training history
    history: Dict[str, List[float]] = {
        'train_loss': [], 'val_loss': [],
        'val_map': [], 'lr': [],
    }

    best_val_loss = float('inf')
    best_map = 0.0
    start_time = time.time()

    print(f"\n{'='*60}")
    print(f"Starting training for {args.epochs} epochs")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss, train_loss_comp = train_one_epoch(
            model, train_loader, processor,
            optimizer, scaler, device,
        )

        # Validate
        val_loss, val_loss_comp = validate(
            model, val_loader, processor, device,
        )

        # Step scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[1]['lr']

        epoch_time = time.time() - epoch_start

        # Evaluate mAP periodically
        val_map = 0.0
        if epoch % args.eval_interval == 0:
            val_map = evaluate_map(
                model, val_loader, processor, device,
                score_threshold=0.3,
            )

        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_map'].append(val_map)
        history['lr'].append(current_lr)

        # Print epoch info
        print(
            f"Epoch [{epoch:3d}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"mAP: {val_map:.4f} | "
            f"LR: {current_lr:.6f} | "
            f"Time: {epoch_time:.1f}s"
        )
        print(
            f"  CE: {val_loss_comp['loss_ce']:.4f} | "
            f"BBox: {val_loss_comp['loss_bbox']:.4f} | "
            f"GIoU: {val_loss_comp['loss_giou']:.4f}"
        )

        # Save best model (by validation loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_args = vars(args).copy()
            save_args['num_classes'] = train_dataset.num_classes
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_map': val_map,
                'args': save_args,
            }
            save_path = os.path.join(args.output_dir, 'best_model.pth')
            torch.save(checkpoint, save_path)
            print(f"  >> New best model saved! Val Loss: {val_loss:.4f}")

        # Save best mAP model separately
        if val_map > best_map:
            best_map = val_map
            save_args = vars(args).copy()
            save_args['num_classes'] = train_dataset.num_classes
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_map': val_map,
                'args': save_args,
            }
            save_path = os.path.join(args.output_dir, 'best_map_model.pth')
            torch.save(checkpoint, save_path)
            print(f"  >> New best mAP model saved! mAP: {val_map:.4f}")

        # Early stopping on validation loss
        if early_stopping(val_loss):
            print(f"\nEarly stopping triggered at epoch {epoch}")
            break

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Total time: {total_time / 60:.1f} minutes")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best validation mAP: {best_map:.4f}")
    print(f"{'='*60}")

    # Save training curves
    plot_training_curves(
        history,
        save_path=os.path.join(args.output_dir, 'training_curves.png'),
    )

    # Save history
    with open(os.path.join(args.output_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    # Save final config
    config = vars(args)
    config['best_val_loss'] = best_val_loss
    config['best_map'] = best_map
    config['total_training_time_min'] = total_time / 60
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)


if __name__ == '__main__':
    main()
