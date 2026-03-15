"""Spectral channel comparison visualization."""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


def visualize_spectral_channels(rgb_image, ir_image, uv_image, xray_image, 
                                layer_masks, pigment_logits, output_path=None):
    """
    Compare spectral channels with model predictions.
    
    Args:
        rgb_image: (3, H, W) RGB image
        ir_image: (1, H, W) Infrared reflectography
        uv_image: (1, H, W) UV fluorescence
        xray_image: (1, H, W) X-ray radiography
        layer_masks: (5, H, W) layer predictions
        pigment_logits: (10, H, W) pigment predictions
        output_path: path to save figure (optional)
    
    Returns:
        fig: matplotlib figure object
    """
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # Row 1: Spectral channels
    # RGB
    rgb_np = rgb_image.cpu().numpy().transpose(1, 2, 0)
    rgb_np = (rgb_np - rgb_np.min()) / (rgb_np.max() - rgb_np.min() + 1e-6)
    
    axes[0, 0].imshow(rgb_np)
    axes[0, 0].set_title('RGB Image', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Infrared
    ir_np = ir_image.cpu().numpy()[0]
    ir_np = (ir_np - ir_np.min()) / (ir_np.max() - ir_np.min() + 1e-6)
    axes[0, 1].imshow(ir_np, cmap='gray')
    axes[0, 1].set_title('Infrared Reflectography', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    # UV
    uv_np = uv_image.cpu().numpy()[0]
    uv_np = (uv_np - uv_np.min()) / (uv_np.max() - uv_np.min() + 1e-6)
    axes[0, 2].imshow(uv_np, cmap='hot')
    axes[0, 2].set_title('UV Fluorescence', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    
    # X-ray
    xray_np = xray_image.cpu().numpy()[0]
    xray_np = (xray_np - xray_np.min()) / (xray_np.max() - xray_np.min() + 1e-6)
    axes[0, 3].imshow(xray_np, cmap='bone')
    axes[0, 3].set_title('X-ray Radiography', fontsize=12, fontweight='bold')
    axes[0, 3].axis('off')
    
    # Row 2: Model predictions
    # Layer segmentation
    layer_probs = F.softmax(layer_masks, dim=0)
    layer_pred = torch.argmax(layer_probs, dim=0).cpu().numpy()
    im1 = axes[1, 0].imshow(layer_pred, cmap='tab10', vmin=0, vmax=4)
    axes[1, 0].set_title('Layer Segmentation', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    plt.colorbar(im1, ax=axes[1, 0], label='Layer')
    
    # Pigment classification
    pigment_probs = F.softmax(pigment_logits, dim=0)
    pigment_pred = torch.argmax(pigment_probs, dim=0).cpu().numpy()
    im2 = axes[1, 1].imshow(pigment_pred, cmap='tab20', vmin=0, vmax=9)
    axes[1, 1].set_title('Pigment Classification', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    plt.colorbar(im2, ax=axes[1, 1], label='Pigment')
    
    # Layer confidence
    layer_conf = torch.max(layer_probs, dim=0)[0].cpu().numpy()
    im3 = axes[1, 2].imshow(layer_conf, cmap='viridis')
    axes[1, 2].set_title('Layer Confidence', fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')
    plt.colorbar(im3, ax=axes[1, 2], label='Confidence')
    
    # Pigment confidence
    pigment_conf = torch.max(pigment_probs, dim=0)[0].cpu().numpy()
    im4 = axes[1, 3].imshow(pigment_conf, cmap='plasma')
    axes[1, 3].set_title('Pigment Confidence', fontsize=12, fontweight='bold')
    axes[1, 3].axis('off')
    plt.colorbar(im4, ax=axes[1, 3], label='Confidence')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Spectral comparison saved to {output_path}")
    
    return fig


def create_spectral_difference_map(rgb_image, ir_image, uv_image, xray_image, output_path=None):
    """
    Create difference maps between spectral channels to highlight hidden structures.
    
    Args:
        rgb_image: (3, H, W) RGB image
        ir_image: (1, H, W) Infrared reflectography
        uv_image: (1, H, W) UV fluorescence
        xray_image: (1, H, W) X-ray radiography
        output_path: path to save figure (optional)
    
    Returns:
        fig: matplotlib figure object
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Normalize all channels
    rgb_np = rgb_image.cpu().numpy().transpose(1, 2, 0)
    rgb_np = (rgb_np - rgb_np.min()) / (rgb_np.max() - rgb_np.min() + 1e-6)
    
    ir_np = ir_image.cpu().numpy()[0]
    ir_np = (ir_np - ir_np.min()) / (ir_np.max() - ir_np.min() + 1e-6)
    
    uv_np = uv_image.cpu().numpy()[0]
    uv_np = (uv_np - uv_np.min()) / (uv_np.max() - uv_np.min() + 1e-6)
    
    xray_np = xray_image.cpu().numpy()[0]
    xray_np = (xray_np - xray_np.min()) / (xray_np.max() - xray_np.min() + 1e-6)
    
    # RGB vs IR difference
    rgb_gray = np.mean(rgb_np, axis=2)
    diff_rgb_ir = np.abs(rgb_gray - ir_np)
    im1 = axes[0, 0].imshow(diff_rgb_ir, cmap='hot')
    axes[0, 0].set_title('RGB vs IR Difference', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # RGB vs UV difference
    diff_rgb_uv = np.abs(rgb_gray - uv_np)
    im2 = axes[0, 1].imshow(diff_rgb_uv, cmap='hot')
    axes[0, 1].set_title('RGB vs UV Difference', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # RGB vs X-ray difference
    diff_rgb_xray = np.abs(rgb_gray - xray_np)
    im3 = axes[0, 2].imshow(diff_rgb_xray, cmap='hot')
    axes[0, 2].set_title('RGB vs X-ray Difference', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    plt.colorbar(im3, ax=axes[0, 2])
    
    # IR vs UV difference
    diff_ir_uv = np.abs(ir_np - uv_np)
    im4 = axes[1, 0].imshow(diff_ir_uv, cmap='hot')
    axes[1, 0].set_title('IR vs UV Difference', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    plt.colorbar(im4, ax=axes[1, 0])
    
    # IR vs X-ray difference
    diff_ir_xray = np.abs(ir_np - xray_np)
    im5 = axes[1, 1].imshow(diff_ir_xray, cmap='hot')
    axes[1, 1].set_title('IR vs X-ray Difference', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    plt.colorbar(im5, ax=axes[1, 1])
    
    # Combined difference (sum of all differences)
    combined_diff = diff_rgb_ir + diff_rgb_uv + diff_rgb_xray + diff_ir_uv + diff_ir_xray
    combined_diff = (combined_diff - combined_diff.min()) / (combined_diff.max() - combined_diff.min() + 1e-6)
    im6 = axes[1, 2].imshow(combined_diff, cmap='hot')
    axes[1, 2].set_title('Combined Spectral Difference', fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')
    plt.colorbar(im6, ax=axes[1, 2])
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Spectral difference map saved to {output_path}")
    
    return fig
