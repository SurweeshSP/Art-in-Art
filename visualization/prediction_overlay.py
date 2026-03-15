"""Prediction overlay visualization."""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def create_prediction_overlay(image, layer_masks, alpha=0.6):
    """
    Overlay layer segmentation predictions on original image.
    
    Args:
        image: (3, H, W) RGB image tensor
        layer_masks: (5, H, W) layer segmentation logits
        alpha: transparency of overlay (0-1)
    
    Returns:
        overlaid_image: (H, W, 3) numpy array
        layer_pred: (H, W) layer assignments
    """
    # Convert to numpy
    image_np = image.cpu().numpy().transpose(1, 2, 0)
    image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min() + 1e-6)
    
    # Get layer predictions
    layer_probs = F.softmax(layer_masks, dim=0)  # (5, H, W)
    layer_pred = torch.argmax(layer_probs, dim=0).cpu().numpy()  # (H, W)
    
    # Create colored layer map
    colors = plt.cm.tab10(np.linspace(0, 1, 5))
    layer_colored = colors[layer_pred][:, :, :3]
    
    # Blend with original
    overlaid = (1 - alpha) * image_np + alpha * layer_colored
    
    return overlaid, layer_pred


def visualize_prediction_overlay(image, layer_masks, pigment_logits, output_path=None):
    """
    Create comprehensive prediction overlay visualization.
    
    Args:
        image: (3, H, W) RGB image tensor
        layer_masks: (5, H, W) layer segmentation logits
        pigment_logits: (10, H, W) pigment classification logits
        output_path: path to save figure (optional)
    
    Returns:
        fig: matplotlib figure object
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Original image
    image_np = image.cpu().numpy().transpose(1, 2, 0)
    image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min() + 1e-6)
    
    axes[0, 0].imshow(image_np)
    axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Layer overlay (alpha=0.3)
    overlaid_light, _ = create_prediction_overlay(image, layer_masks, alpha=0.3)
    axes[0, 1].imshow(overlaid_light)
    axes[0, 1].set_title('Layer Overlay (α=0.3)', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Layer overlay (alpha=0.6)
    overlaid_medium, _ = create_prediction_overlay(image, layer_masks, alpha=0.6)
    axes[0, 2].imshow(overlaid_medium)
    axes[0, 2].set_title('Layer Overlay (α=0.6)', fontsize=14, fontweight='bold')
    axes[0, 2].axis('off')
    
    # Layer segmentation map
    layer_probs = F.softmax(layer_masks, dim=0)
    layer_pred = torch.argmax(layer_probs, dim=0).cpu().numpy()
    im1 = axes[1, 0].imshow(layer_pred, cmap='tab10', vmin=0, vmax=4)
    axes[1, 0].set_title('Layer Segmentation Map', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    plt.colorbar(im1, ax=axes[1, 0], label='Layer ID')
    
    # Layer confidence
    layer_conf = torch.max(layer_probs, dim=0)[0].cpu().numpy()
    im2 = axes[1, 1].imshow(layer_conf, cmap='viridis')
    axes[1, 1].set_title('Layer Prediction Confidence', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    plt.colorbar(im2, ax=axes[1, 1], label='Confidence')
    
    # Pigment segmentation
    pigment_probs = F.softmax(pigment_logits, dim=0)
    pigment_pred = torch.argmax(pigment_probs, dim=0).cpu().numpy()
    im3 = axes[1, 2].imshow(pigment_pred, cmap='tab20', vmin=0, vmax=9)
    axes[1, 2].set_title('Pigment Classification Map', fontsize=14, fontweight='bold')
    axes[1, 2].axis('off')
    plt.colorbar(im3, ax=axes[1, 2], label='Pigment ID')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Prediction overlay saved to {output_path}")
    
    return fig
