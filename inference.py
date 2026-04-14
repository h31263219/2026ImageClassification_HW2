"""Inference script for digit detection using DETR.

Generates predictions on the test set and outputs pred.json
in COCO format for CodaBench submission.

Usage:
    python inference.py --data_dir ./nycu-hw2-data \
        --checkpoint ./output/best_map_model.pth
"""

import argparse
import json
import os
import zipfile
from typing import Dict, List

import torch
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DetrImageProcessor

from dataset import DetectionTransform, TestDataset, collate_fn
from model import build_model


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description='DETR digit detection inference',
    )
    parser.add_argument(
        '--data_dir', type=str,
        default='./nycu-hw2-data',
        help='Path to dataset root directory',
    )
    parser.add_argument(
        '--checkpoint', type=str,
        default='./output/best_map_model.pth',
        help='Path to model checkpoint',
    )
    parser.add_argument(
        '--output_dir', type=str, default='./output',
        help='Directory to save predictions',
    )
    parser.add_argument(
        '--batch_size', type=int, default=16,
        help='Inference batch size',
    )
    parser.add_argument(
        '--num_workers', type=int, default=0,
        help='Number of data loading workers',
    )
    parser.add_argument(
        '--score_threshold', type=float, default=0.1,
        help='Minimum detection score threshold',
    )
    parser.add_argument(
        '--image_size', type=int, default=480,
        help='Resize shortest side to this value',
    )
    parser.add_argument(
        '--student_id', type=str, default='314560017',
        help='Student ID for submission file naming',
    )
    return parser.parse_args()


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    score_threshold: float = 0.1,
) -> List[Dict]:
    """Run inference on the test set.

    Args:
        model: Trained DETR model.
        dataloader: Test data loader.
        device: Target device.
        score_threshold: Minimum score to keep detection.

    Returns:
        List of prediction dictionaries in COCO format.
    """
    model.eval()
    all_predictions = []

    for images, infos in tqdm(dataloader, desc='Inference'):
        # Pad images to same size
        max_h = max(img.shape[1] for img in images)
        max_w = max(img.shape[2] for img in images)

        batch_size = len(images)
        padded = torch.zeros(batch_size, 3, max_h, max_w)
        pixel_mask = torch.zeros(
            batch_size, max_h, max_w, dtype=torch.long,
        )

        for i, img in enumerate(images):
            _, h, w = img.shape
            padded[i, :, :h, :w] = img
            pixel_mask[i, :h, :w] = 1

        padded = padded.to(device)
        pixel_mask = pixel_mask.to(device)

        with autocast('cuda'):
            outputs = model(
                pixel_values=padded,
                pixel_mask=pixel_mask,
            )

        # Post-process
        probs = outputs.logits.softmax(-1)
        scores, labels = probs[..., :-1].max(-1)

        for i in range(batch_size):
            img_id = infos[i]['image_id'].item()
            orig_h, orig_w = infos[i]['orig_size'].tolist()

            # Filter by score
            keep = scores[i] > score_threshold
            pred_scores = scores[i][keep]
            pred_labels = labels[i][keep]
            pred_boxes = outputs.pred_boxes[i][keep]

            if len(pred_boxes) > 0:
                pred_boxes = pred_boxes.cpu()
                # Convert from normalized [cx, cy, w, h] to
                # absolute [x_min, y_min, w, h]
                cx = pred_boxes[:, 0] * orig_w
                cy = pred_boxes[:, 1] * orig_h
                bw = pred_boxes[:, 2] * orig_w
                bh = pred_boxes[:, 3] * orig_h
                x_min = cx - bw / 2
                y_min = cy - bh / 2

                for j in range(len(pred_scores)):
                    # category_id is 1-indexed
                    cat_id = pred_labels[j].item() + 1
                    all_predictions.append({
                        'image_id': img_id,
                        'bbox': [
                            round(x_min[j].item(), 6),
                            round(y_min[j].item(), 6),
                            round(bw[j].item(), 6),
                            round(bh[j].item(), 6),
                        ],
                        'score': round(pred_scores[j].item(), 6),
                        'category_id': cat_id,
                    })

    return all_predictions


def main() -> None:
    """Main inference pipeline."""
    args = parse_args()

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(
        args.checkpoint, map_location=device, weights_only=False,
    )
    saved_args = checkpoint.get('args', {})

    # Build model
    num_classes = saved_args.get('num_classes', 10)

    # Detect num_queries from checkpoint state_dict
    num_queries = 100
    if 'model.query_position_embeddings.weight' in checkpoint[
        'model_state_dict'
    ]:
        num_queries = checkpoint[
            'model_state_dict'
        ]['model.query_position_embeddings.weight'].shape[0]

    model = build_model(
        num_classes=num_classes,
        pretrained=False,
        num_queries=num_queries,
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Loaded model from epoch {checkpoint.get('epoch', '?')}")
    print(f"Checkpoint val_loss: {checkpoint.get('val_loss', '?')}")
    print(f"Checkpoint val_map: {checkpoint.get('val_map', '?')}")

    # Test dataset
    test_img_dir = os.path.join(args.data_dir, 'test')
    test_transform = DetectionTransform(
        size=args.image_size, train=False,
    )
    test_dataset = TestDataset(
        img_dir=test_img_dir,
        transforms=test_transform,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    print(f"Test images: {len(test_dataset)}")

    # Run inference
    predictions = run_inference(
        model, test_loader, device,
        score_threshold=args.score_threshold,
    )

    print(f"Total predictions: {len(predictions)}")

    # Save pred.json
    pred_path = os.path.join(args.output_dir, 'pred.json')
    with open(pred_path, 'w') as f:
        json.dump(predictions, f, indent=2)
    print(f"Predictions saved to {pred_path}")

    # Create submission zip
    zip_name = f"{args.student_id}_HW2_submission.zip"
    zip_path = os.path.join(args.output_dir, zip_name)
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(pred_path, 'pred.json')
    print(f"Submission zip saved to {zip_path}")

    # Summary statistics
    if predictions:
        scores = [p['score'] for p in predictions]
        cats = [p['category_id'] for p in predictions]
        print(f"\nPrediction Statistics:")
        print(f"  Score range: [{min(scores):.4f}, {max(scores):.4f}]")
        print(f"  Mean score: {sum(scores)/len(scores):.4f}")
        print(f"  Predictions per category:")
        for cat_id in sorted(set(cats)):
            count = cats.count(cat_id)
            print(f"    Category {cat_id}: {count}")


if __name__ == '__main__':
    main()
