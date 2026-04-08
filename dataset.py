"""Dataset module for digit detection using DETR.

Provides a COCO-format dataset class and image transforms
for training/validation of the DETR object detection model.
"""

import os
from typing import Any, Dict, List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
import json


class CocoDetectionDataset(Dataset):
    """COCO-format object detection dataset.

    Loads images and annotations from a COCO-format JSON file,
    converting bounding boxes to the format expected by DETR
    (normalized [cx, cy, w, h]).

    Args:
        img_dir: Path to directory containing images.
        ann_file: Path to COCO-format annotation JSON file.
        transforms: Optional image/target transforms.
    """

    def __init__(
        self,
        img_dir: str,
        ann_file: str,
        transforms: Any = None,
    ) -> None:
        self.img_dir = img_dir
        self.transforms = transforms

        with open(ann_file, 'r') as f:
            coco_data = json.load(f)

        self.images = {img['id']: img for img in coco_data['images']}
        self.image_ids = list(self.images.keys())

        # Group annotations by image_id
        self.img_annotations: Dict[int, List[Dict]] = {
            img_id: [] for img_id in self.image_ids
        }
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id in self.img_annotations:
                self.img_annotations[img_id].append(ann)

        # Category mapping: original category_id (1-10) -> 0-indexed (0-9)
        self.categories = coco_data.get('categories', [])
        self.cat_id_to_label = {
            cat['id']: idx for idx, cat in enumerate(self.categories)
        }
        self.num_classes = len(self.categories)

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(
        self, idx: int,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Get an image and its annotations.

        Returns:
            Tuple of (image_tensor, target_dict) where target_dict contains:
                - labels: (N,) tensor of class labels (0-indexed)
                - boxes: (N, 4) tensor of normalized [cx, cy, w, h] boxes
                - image_id: scalar tensor of image ID
                - orig_size: (2,) tensor of original [h, w]
        """
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]

        # Load image
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        image = Image.open(img_path).convert('RGB')

        w, h = image.size
        annotations = self.img_annotations[img_id]

        # Parse annotations
        boxes = []
        labels = []
        for ann in annotations:
            x_min, y_min, bw, bh = ann['bbox']
            # Convert [x_min, y_min, w, h] -> [cx, cy, w, h] normalized
            cx = (x_min + bw / 2.0) / w
            cy = (y_min + bh / 2.0) / h
            bw_norm = bw / w
            bh_norm = bh / h
            boxes.append([cx, cy, bw_norm, bh_norm])
            labels.append(self.cat_id_to_label[ann['category_id']])

        target = {
            'labels': torch.tensor(labels, dtype=torch.long),
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'image_id': torch.tensor([img_id]),
            'orig_size': torch.tensor([h, w]),
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target


class TestDataset(Dataset):
    """Test dataset for inference (no annotations).

    Args:
        img_dir: Path to directory containing test images.
        transforms: Image transforms to apply.
    """

    def __init__(
        self,
        img_dir: str,
        transforms: Any = None,
    ) -> None:
        self.img_dir = img_dir
        self.transforms = transforms
        self.image_files = sorted(
            os.listdir(img_dir),
            key=lambda x: int(os.path.splitext(x)[0]),
        )

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(
        self, idx: int,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Get a test image.

        Returns:
            Tuple of (image_tensor, info_dict) with image_id and orig_size.
        """
        filename = self.image_files[idx]
        img_id = int(os.path.splitext(filename)[0])
        img_path = os.path.join(self.img_dir, filename)
        image = Image.open(img_path).convert('RGB')
        w, h = image.size

        info = {
            'image_id': torch.tensor([img_id]),
            'orig_size': torch.tensor([h, w]),
        }

        if self.transforms is not None:
            image, info = self.transforms(image, info)

        return image, info


class DetectionTransform:
    """Transform pipeline for DETR training/validation.

    Resizes images and applies normalization. For training,
    also applies data augmentation (random horizontal flip,
    color jitter, random resize).

    Args:
        size: Target image size (shortest side).
        max_size: Maximum image size (longest side).
        train: Whether to apply training augmentations.
    """

    def __init__(
        self,
        size: int = 800,
        max_size: int = 1333,
        train: bool = False,
    ) -> None:
        self.size = size
        self.max_size = max_size
        self.train = train

        self.normalize = T.Compose([
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __call__(
        self,
        image: Image.Image,
        target: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Apply transforms to image and target.

        Args:
            image: PIL Image.
            target: Target dictionary with boxes and labels.

        Returns:
            Tuple of (transformed_image, transformed_target).
        """
        if self.train:
            # Random horizontal flip
            if torch.rand(1).item() < 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                if 'boxes' in target and len(target['boxes']) > 0:
                    boxes = target['boxes'].clone()
                    # Flip cx: cx -> 1 - cx (boxes are normalized)
                    boxes[:, 0] = 1.0 - boxes[:, 0]
                    target['boxes'] = boxes

            # Random color jitter
            color_jitter = T.ColorJitter(
                brightness=0.3, contrast=0.3,
                saturation=0.3, hue=0.1,
            )
            image = color_jitter(image)

        # Resize
        image = T.Resize(
            self.size, max_size=self.max_size,
        )(image)

        # Normalize
        image = self.normalize(image)

        return image, target


def collate_fn(
    batch: List[Tuple[torch.Tensor, Dict[str, Any]]],
) -> Tuple[List[torch.Tensor], List[Dict[str, Any]]]:
    """Custom collate function for detection datasets.

    Since images may have different sizes and different numbers
    of objects, we cannot simply stack them. Instead, return
    lists of images and targets.

    Args:
        batch: List of (image, target) tuples.

    Returns:
        Tuple of (images_list, targets_list).
    """
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets
