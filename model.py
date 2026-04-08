"""DETR model for digit detection.

Builds a DETR (DEtection TRansformer) model with ResNet-50 backbone
for detecting digits (0-9) in images. Uses HuggingFace's
pretrained DETR model and replaces the classification head.
"""

import torch
import torch.nn as nn
from transformers import DetrForObjectDetection, DetrConfig


def build_model(
    num_classes: int = 10,
    pretrained: bool = True,
    num_queries: int = 50,
) -> nn.Module:
    """Build a DETR model for digit detection.

    Uses facebook/detr-resnet-50 pretrained on COCO as initialization,
    then replaces the classification head for the target number of classes.

    Args:
        num_classes: Number of object classes (default 10 for digits).
        pretrained: Whether to use pretrained COCO weights.
        num_queries: Number of object queries for DETR decoder.

    Returns:
        DetrForObjectDetection model.
    """
    if pretrained:
        # Load pretrained DETR (91 COCO classes)
        model = DetrForObjectDetection.from_pretrained(
            'facebook/detr-resnet-50',
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )
    else:
        config = DetrConfig(
            num_labels=num_classes,
            num_queries=num_queries,
        )
        model = DetrForObjectDetection(config)

    return model


def get_model_info(model: nn.Module) -> dict:
    """Get model parameter information.

    Args:
        model: The DETR model.

    Returns:
        Dictionary with parameter counts.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    backbone_params = sum(
        p.numel()
        for name, p in model.named_parameters()
        if 'backbone' in name
    )
    transformer_params = sum(
        p.numel()
        for name, p in model.named_parameters()
        if 'encoder' in name or 'decoder' in name
    )

    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'backbone_params': backbone_params,
        'transformer_params': transformer_params,
    }
