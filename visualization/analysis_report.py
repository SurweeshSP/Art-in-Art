"""Comprehensive analysis report generation."""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def generate_analysis_report(image, layer_masks, pigment_logits, reconstructed=None, 
                            output_path='analysis_report.png'):
    """
    Generate comprehensive analysis report with all visualizations.
    
    Args:
        image: (3, H, W) RGB image
        layer_masks: (5, H, W) layer predictions
        pigment_logits: (10, H, W) pigment predictions
        reconstructed: (3, H, W) reconstructed hidden image (optional)
        output_path: path to save figure
    
    Returns:
        fig: matplotlib figure object
    """
    fig = plt.figure(figsize=(24, 16))
    gs = GridSpec(4, 4, figure=fig, hspace=0.35, wspace=0.3)
    
    # Normalize image
    image_np = image.cpu().numpy().transpose(1, 2, 0)
    image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min() + 1e-6)
    
    # Get predictions
    layer_probs = F.softmax(layer_masks, dim=0)
    layer_pred = torch.argmax(layer_probs, dim=0).cpu().numpy()
    layer_conf = torch.max(layer_probs, dim=0)[0].cpu().numpy()
    
    pigment_probs = F.softmax(pigment_logits, dim=0)
    pigment_pred = torch.argmax(pigment_probs, dim=0).cpu().numpy()
    pigment_conf = torch.max(pigment_probs, dim=0)[0].cpu().numpy()
    
    # Row 1: Input and basic predictions
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(image_np)
    ax1.set_title('Original Image', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    im = ax2.imshow(layer_pred, cmap='tab10', vmin=0, vmax=4)
    ax2.set_title('Layer Segmentation', fontsize=12, fontweight='bold')
    ax2.axis('off')
    plt.colorbar(im, ax=ax2, label='Layer')
    
    ax3 = fig.add_subplot(gs[0, 2])
    im = ax3.imshow(pigment_pred, cmap='tab20', vmin=0, vmax=9)
    ax3.set_title('Pigment Classification', fontsize=12, fontweight='bold')
    ax3.axis('off')
    plt.colorbar(im, ax=ax3, label='Pigment')
    
    ax4 = fig.add_subplot(gs[0, 3])
    im = ax4.imshow(layer_conf, cmap='viridis')
    ax4.set_title('Layer Confidence', fontsize=12, fontweight='bold')
    ax4.axis('off')
    plt.colorbar(im, ax=ax4)
    
    # Row 2: Confidence and uncertainty
    ax5 = fig.add_subplot(gs[1, 0])
    im = ax5.imshow(pigment_conf, cmap='plasma')
    ax5.set_title('Pigment Confidence', fontsize=12, fontweight='bold')
    ax5.axis('off')
    plt.colorbar(im, ax=ax5)
    
    # Layer entropy
    ax6 = fig.add_subplot(gs[1, 1])
    layer_entropy = -torch.sum(layer_probs * torch.log(layer_probs + 1e-6), dim=0).cpu().numpy()
    im = ax6.imshow(layer_entropy, cmap='hot')
    ax6.set_title('Layer Uncertainty', fontsize=12, fontweight='bold')
    ax6.axis('off')
    plt.colorbar(im, ax=ax6)
    
    # Pigment entropy
    ax7 = fig.add_subplot(gs[1, 2])
    pigment_entropy = -torch.sum(pigment_probs * torch.log(pigment_probs + 1e-6), dim=0).cpu().numpy()
    im = ax7.imshow(pigment_entropy, cmap='hot')
    ax7.set_title('Pigment Uncertainty', fontsize=12, fontweight='bold')
    ax7.axis('off')
    plt.colorbar(im, ax=ax7)
    
    # Combined attention
    ax8 = fig.add_subplot(gs[1, 3])
    combined_attention = (layer_conf + pigment_conf) / 2
    im = ax8.imshow(combined_attention, cmap='viridis')
    ax8.set_title('Combined Attention', fontsize=12, fontweight='bold')
    ax8.axis('off')
    plt.colorbar(im, ax=ax8)
    
    # Row 3: Overlays and heatmaps
    ax9 = fig.add_subplot(gs[2, 0])
    overlay_light = 0.7 * image_np + 0.3 * plt.cm.tab10(layer_pred / 4)[:, :, :3]
    ax9.imshow(overlay_light)
    ax9.set_title('Layer Overlay (α=0.3)', fontsize=12, fontweight='bold')
    ax9.axis('off')
    
    ax10 = fig.add_subplot(gs[2, 1])
    overlay_medium = 0.5 * image_np + 0.5 * plt.cm.tab10(layer_pred / 4)[:, :, :3]
    ax10.imshow(overlay_medium)
    ax10.set_title('Layer Overlay (α=0.5)', fontsize=12, fontweight='bold')
    ax10.axis('off')
    
    ax11 = fig.add_subplot(gs[2, 2])
    ax11.imshow(image_np)
    ax11.imshow(layer_conf, cmap='jet', alpha=0.5)
    ax11.set_title('Layer Confidence Heatmap', fontsize=12, fontweight='bold')
    ax11.axis('off')
    
    ax12 = fig.add_subplot(gs[2, 3])
    ax12.imshow(image_np)
    ax12.imshow(pigment_conf, cmap='hot', alpha=0.5)
    ax12.set_title('Pigment Confidence Heatmap', fontsize=12, fontweight='bold')
    ax12.axis('off')
    
    # Row 4: Statistics and reconstruction
    ax13 = fig.add_subplot(gs[3, 0])
    layer_coverage = [(layer_pred == i).sum() / layer_pred.size for i in range(5)]
    ax13.bar(range(5), layer_coverage, color=plt.cm.tab10(np.arange(5) / 4))
    ax13.set_title('Layer Coverage', fontsize=12, fontweight='bold')
    ax13.set_xlabel('Layer ID')
    ax13.set_ylabel('Coverage %')
    ax13.set_ylim([0, 1])
    ax13.grid(alpha=0.3, axis='y')
    
    ax14 = fig.add_subplot(gs[3, 1])
    pigment_coverage = [(pigment_pred == i).sum() / pigment_pred.size for i in range(10)]
    ax14.bar(range(10), pigment_coverage, color=plt.cm.tab20(np.arange(10) / 10))
    ax14.set_title('Pigment Coverage', fontsize=12, fontweight='bold')
    ax14.set_xlabel('Pigment ID')
    ax14.set_ylabel('Coverage %')
    ax14.set_ylim([0, 1])
    ax14.grid(alpha=0.3, axis='y')
    
    # Reconstruction if available
    if reconstructed is not None:
        ax15 = fig.add_subplot(gs[3, 2])
        reconstructed_np = reconstructed.cpu().numpy().transpose(1, 2, 0)
        ax15.imshow(reconstructed_np)
        ax15.set_title('Reconstructed Hidden Image', fontsize=12, fontweight='bold')
        ax15.axis('off')
        
        ax16 = fig.add_subplot(gs[3, 3])
        difference = np.abs(reconstructed_np - image_np)
        im = ax16.imshow(difference)
        ax16.set_title('Reconstruction Difference', fontsize=12, fontweight='bold')
        ax16.axis('off')
        plt.colorbar(im, ax=ax16)
    else:
        # Show additional statistics
        ax15 = fig.add_subplot(gs[3, 2])
        layer_conf_dist = layer_conf.flatten()
        ax15.hist(layer_conf_dist, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        ax15.set_title('Layer Confidence Distribution', fontsize=12, fontweight='bold')
        ax15.set_xlabel('Confidence')
        ax15.set_ylabel('Frequency')
        ax15.grid(alpha=0.3, axis='y')
        
        ax16 = fig.add_subplot(gs[3, 3])
        pigment_conf_dist = pigment_conf.flatten()
        ax16.hist(pigment_conf_dist, bins=30, edgecolor='black', alpha=0.7, color='coral')
        ax16.set_title('Pigment Confidence Distribution', fontsize=12, fontweight='bold')
        ax16.set_xlabel('Confidence')
        ax16.set_ylabel('Frequency')
        ax16.grid(alpha=0.3, axis='y')
    
    # Add title
    fig.suptitle('Comprehensive Painting Analysis Report', fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Analysis report saved to {output_path}")
    
    return fig


def generate_statistics_summary(layer_masks, pigment_logits, output_path=None):
    """
    Generate detailed statistics summary.
    
    Args:
        layer_masks: (5, H, W) layer predictions
        pigment_logits: (10, H, W) pigment predictions
        output_path: path to save figure (optional)
    
    Returns:
        fig: matplotlib figure object
        stats: dictionary of statistics
    """
    layer_probs = F.softmax(layer_masks, dim=0)
    pigment_probs = F.softmax(pigment_logits, dim=0)
    
    layer_pred = torch.argmax(layer_probs, dim=0).cpu().numpy()
    pigment_pred = torch.argmax(pigment_probs, dim=0).cpu().numpy()
    
    # Calculate statistics
    stats = {}
    
    # Layer statistics
    layer_names = ['Layer 0', 'Layer 1', 'Layer 2', 'Layer 3', 'Layer 4']
    pigment_names = [
        'Lead White', 'Ultramarine', 'Ochre', 'Vermillion', 'Azurite',
        'Umber', 'Charcoal', 'Lapis Lazuli', 'Cadmium Yellow', 'Titanium White'
    ]
    
    for i in range(5):
        coverage = (layer_pred == i).sum() / layer_pred.size
        confidence = layer_probs[i].mean().item()
        stats[f'layer_{i}_coverage'] = coverage
        stats[f'layer_{i}_confidence'] = confidence
    
    # Pigment statistics
    for i in range(10):
        coverage = (pigment_pred == i).sum() / pigment_pred.size
        confidence = pigment_probs[i].mean().item()
        stats[f'pigment_{i}_coverage'] = coverage
        stats[f'pigment_{i}_confidence'] = confidence
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Layer coverage
    layer_coverage = [stats[f'layer_{i}_coverage'] for i in range(5)]
    axes[0, 0].bar(layer_names, layer_coverage, color=plt.cm.tab10(np.arange(5) / 4))
    axes[0, 0].set_title('Layer Coverage', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Coverage Ratio')
    axes[0, 0].grid(alpha=0.3, axis='y')
    
    # Layer confidence
    layer_conf = [stats[f'layer_{i}_confidence'] for i in range(5)]
    axes[0, 1].bar(layer_names, layer_conf, color=plt.cm.tab10(np.arange(5) / 4))
    axes[0, 1].set_title('Layer Prediction Confidence', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Mean Confidence')
    axes[0, 1].set_ylim([0, 1])
    axes[0, 1].grid(alpha=0.3, axis='y')
    
    # Pigment coverage
    pigment_coverage = [stats[f'pigment_{i}_coverage'] for i in range(10)]
    axes[1, 0].bar(range(10), pigment_coverage, color=plt.cm.tab20(np.arange(10) / 10))
    axes[1, 0].set_title('Pigment Coverage', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Pigment ID')
    axes[1, 0].set_ylabel('Coverage Ratio')
    axes[1, 0].set_xticks(range(10))
    axes[1, 0].grid(alpha=0.3, axis='y')
    
    # Pigment confidence
    pigment_conf = [stats[f'pigment_{i}_confidence'] for i in range(10)]
    axes[1, 1].bar(range(10), pigment_conf, color=plt.cm.tab20(np.arange(10) / 10))
    axes[1, 1].set_title('Pigment Prediction Confidence', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Pigment ID')
    axes[1, 1].set_ylabel('Mean Confidence')
    axes[1, 1].set_xticks(range(10))
    axes[1, 1].set_ylim([0, 1])
    axes[1, 1].grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Statistics summary saved to {output_path}")
    
    return fig, stats
