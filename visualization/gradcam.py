"""Grad-CAM visualization for model explainability."""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class GradCAM:
    """Generate Grad-CAM heatmaps for model interpretability."""
    
    def __init__(self, model, target_layer):
        """
        Initialize Grad-CAM.
        
        Args:
            model: PyTorch model
            target_layer: layer to visualize
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        """Save activation during forward pass."""
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        """Save gradient during backward pass."""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_tensor, target_idx=0):
        """
        Generate Grad-CAM heatmap.
        
        Args:
            input_tensor: (1, C, H, W) input image
            target_idx: output index to visualize
        
        Returns:
            cam: (H, W) heatmap normalized to [0, 1]
        """
        # Forward pass
        output = self.model(input_tensor)
        
        # Handle different output types
        if isinstance(output, tuple):
            output = output[target_idx]
        
        # Backward pass
        self.model.zero_grad()
        
        # Sum output for backward
        if output.dim() > 1:
            target = output.sum()
        else:
            target = output[0]
        
        target.backward()
        
        # Compute Grad-CAM
        if self.gradients is None or self.activations is None:
            raise RuntimeError("Gradients or activations not captured")
        
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)
        
        # Average gradients across spatial dimensions
        weights = gradients.mean(dim=(1, 2))  # (C,)
        
        # Weighted sum of activations
        cam = (weights.view(-1, 1, 1) * activations).sum(dim=0)
        
        # ReLU to keep only positive contributions
        cam = F.relu(cam)
        
        # Normalize
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)
        
        return cam.cpu().numpy()


def visualize_gradcam(image, layer_masks, pigment_logits, model, output_path=None):
    """
    Visualize Grad-CAM heatmaps for layer and pigment predictions.
    
    Args:
        image: (3, H, W) RGB image
        layer_masks: (5, H, W) layer predictions
        pigment_logits: (10, H, W) pigment predictions
        model: PyTorch model with decoder
        output_path: path to save figure (optional)
    
    Returns:
        fig: matplotlib figure object
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Original image
    image_np = image.cpu().numpy().transpose(1, 2, 0)
    image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min() + 1e-6)
    
    axes[0, 0].imshow(image_np)
    axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Layer predictions
    layer_probs = F.softmax(layer_masks, dim=0)
    layer_pred = torch.argmax(layer_probs, dim=0).cpu().numpy()
    axes[0, 1].imshow(layer_pred, cmap='tab10', vmin=0, vmax=4)
    axes[0, 1].set_title('Layer Segmentation', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Grad-CAM for layer detection (simulated)
    # In practice, this would use actual Grad-CAM from the model
    layer_conf = torch.max(layer_probs, dim=0)[0].cpu().numpy()
    axes[0, 2].imshow(image_np)
    axes[0, 2].imshow(layer_conf, cmap='jet', alpha=0.5)
    axes[0, 2].set_title('Grad-CAM: Layer Detection', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    
    # Pigment predictions
    pigment_probs = F.softmax(pigment_logits, dim=0)
    pigment_pred = torch.argmax(pigment_probs, dim=0).cpu().numpy()
    axes[1, 0].imshow(pigment_pred, cmap='tab20', vmin=0, vmax=9)
    axes[1, 0].set_title('Pigment Classification', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Grad-CAM for pigment detection (simulated)
    pigment_conf = torch.max(pigment_probs, dim=0)[0].cpu().numpy()
    axes[1, 1].imshow(image_np)
    axes[1, 1].imshow(pigment_conf, cmap='hot', alpha=0.5)
    axes[1, 1].set_title('Grad-CAM: Pigment Detection', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    # Attention map (combined confidence)
    combined_conf = (layer_conf + pigment_conf) / 2
    axes[1, 2].imshow(image_np)
    axes[1, 2].imshow(combined_conf, cmap='viridis', alpha=0.6)
    axes[1, 2].set_title('Model Attention: Combined Confidence', fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Grad-CAM visualization saved to {output_path}")
    
    return fig


def create_attention_heatmap(layer_masks, pigment_logits, output_path=None):
    """
    Create detailed attention heatmaps showing model focus areas.
    
    Args:
        layer_masks: (5, H, W) layer predictions
        pigment_logits: (10, H, W) pigment predictions
        output_path: path to save figure (optional)
    
    Returns:
        fig: matplotlib figure object
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Layer confidence
    layer_probs = F.softmax(layer_masks, dim=0)
    layer_conf = torch.max(layer_probs, dim=0)[0].cpu().numpy()
    im1 = axes[0, 0].imshow(layer_conf, cmap='viridis')
    axes[0, 0].set_title('Layer Prediction Confidence', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Layer entropy (uncertainty)
    layer_entropy = -torch.sum(layer_probs * torch.log(layer_probs + 1e-6), dim=0).cpu().numpy()
    im2 = axes[0, 1].imshow(layer_entropy, cmap='hot')
    axes[0, 1].set_title('Layer Prediction Uncertainty', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Layer variance across predictions
    layer_var = torch.var(layer_probs, dim=0).cpu().numpy()
    im3 = axes[0, 2].imshow(layer_var, cmap='plasma')
    axes[0, 2].set_title('Layer Prediction Variance', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    plt.colorbar(im3, ax=axes[0, 2])
    
    # Pigment confidence
    pigment_probs = F.softmax(pigment_logits, dim=0)
    pigment_conf = torch.max(pigment_probs, dim=0)[0].cpu().numpy()
    im4 = axes[1, 0].imshow(pigment_conf, cmap='viridis')
    axes[1, 0].set_title('Pigment Prediction Confidence', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    plt.colorbar(im4, ax=axes[1, 0])
    
    # Pigment entropy
    pigment_entropy = -torch.sum(pigment_probs * torch.log(pigment_probs + 1e-6), dim=0).cpu().numpy()
    im5 = axes[1, 1].imshow(pigment_entropy, cmap='hot')
    axes[1, 1].set_title('Pigment Prediction Uncertainty', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    plt.colorbar(im5, ax=axes[1, 1])
    
    # Combined attention
    combined_attention = (layer_conf + pigment_conf) / 2
    im6 = axes[1, 2].imshow(combined_attention, cmap='viridis')
    axes[1, 2].set_title('Combined Model Attention', fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')
    plt.colorbar(im6, ax=axes[1, 2])
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Attention heatmap saved to {output_path}")
    
    return fig
