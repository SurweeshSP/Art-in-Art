"""Hidden image reconstruction demonstration."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class SimpleHiddenImageReconstructor(nn.Module):
    """
    Simple autoencoder for hidden image reconstruction.
    Encodes spectral information → latent space → reconstructs hidden image.
    """
    
    def __init__(self, in_channels=4, latent_dim=128):
        """
        Initialize reconstructor.
        
        Args:
            in_channels: number of input spectral channels (RGB + IR)
            latent_dim: dimension of latent representation
        """
        super().__init__()
        
        # Encoder: Spectral channels → latent representation
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.fc_encode = nn.Linear(128, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 128 * 14 * 14)
        
        # Decoder: Latent space → hidden image
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid()  # Output in [0, 1]
        )
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: (B, in_channels, H, W) input
        
        Returns:
            reconstructed: (B, 3, H, W) reconstructed image
            latent: (B, latent_dim) latent representation
        """
        # Encode
        encoded = self.encoder(x)
        encoded = encoded.view(encoded.size(0), -1)
        latent = self.fc_encode(encoded)
        
        # Decode
        decoded = self.fc_decode(latent)
        decoded = decoded.view(decoded.size(0), 128, 14, 14)
        reconstructed = self.decoder(decoded)
        
        return reconstructed, latent


def demo_hidden_image_reconstruction(output_path=None):
    """
    Demonstrate hidden image reconstruction pipeline.
    
    Args:
        output_path: path to save figure (optional)
    
    Returns:
        reconstructor: trained reconstructor model
        fig: matplotlib figure object
    """
    # Initialize reconstructor
    reconstructor = SimpleHiddenImageReconstructor(in_channels=4, latent_dim=128)
    
    # Simulate multispectral input (RGB + IR)
    rgb = torch.randn(1, 3, 224, 224)
    ir = torch.randn(1, 1, 224, 224)
    multispectral = torch.cat([rgb, ir], dim=1)  # (1, 4, 224, 224)
    
    # Reconstruct hidden image
    reconstructed, latent = reconstructor(multispectral)
    
    # Resize reconstructed to match input size
    reconstructed = F.interpolate(reconstructed, size=(224, 224), mode='bilinear', align_corners=False)
    
    print(f"Input shape: {multispectral.shape}")
    print(f"Latent representation: {latent.shape}")
    print(f"Reconstructed image: {reconstructed.shape}")
    
    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Input channels
    rgb_np = rgb[0].cpu().numpy().transpose(1, 2, 0)
    rgb_np = (rgb_np - rgb_np.min()) / (rgb_np.max() - rgb_np.min() + 1e-6)
    axes[0, 0].imshow(rgb_np)
    axes[0, 0].set_title('Input: RGB Image', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    ir_np = ir[0, 0].cpu().numpy()
    ir_np = (ir_np - ir_np.min()) / (ir_np.max() - ir_np.min() + 1e-6)
    axes[0, 1].imshow(ir_np, cmap='gray')
    axes[0, 1].set_title('Input: Infrared Channel', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Latent space visualization
    latent_np = latent[0].cpu().detach().numpy()
    axes[0, 2].bar(range(min(50, len(latent_np))), latent_np[:50])
    axes[0, 2].set_title('Latent Representation (first 50 dims)', fontsize=12, fontweight='bold')
    axes[0, 2].set_xlabel('Dimension')
    axes[0, 2].set_ylabel('Value')
    axes[0, 2].grid(alpha=0.3)
    
    # Reconstruction
    reconstructed_np = reconstructed[0].cpu().detach().numpy().transpose(1, 2, 0)
    axes[1, 0].imshow(reconstructed_np)
    axes[1, 0].set_title('Reconstructed Hidden Image', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Difference map (what was hidden)
    difference = np.abs(reconstructed_np - rgb_np)
    axes[1, 1].imshow(difference)
    axes[1, 1].set_title('Difference Map (Hidden Content)', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    # Confidence/uncertainty
    uncertainty = np.std([reconstructed_np, rgb_np], axis=0).mean(axis=2)
    im = axes[1, 2].imshow(uncertainty, cmap='hot')
    axes[1, 2].set_title('Reconstruction Uncertainty', fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')
    plt.colorbar(im, ax=axes[1, 2])
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Reconstruction demo saved to {output_path}")
    
    return reconstructor, fig


def visualize_reconstruction_stages(rgb_image, ir_image, reconstructor, output_path=None):
    """
    Visualize different stages of reconstruction.
    
    Args:
        rgb_image: (3, H, W) RGB image
        ir_image: (1, H, W) Infrared image
        reconstructor: trained reconstructor model
        output_path: path to save figure (optional)
    
    Returns:
        fig: matplotlib figure object
    """
    # Prepare input
    multispectral = torch.cat([rgb_image.unsqueeze(0), ir_image.unsqueeze(0)], dim=1)
    
    # Get reconstruction
    reconstructed, latent = reconstructor(multispectral)
    
    # Resize reconstructed to match input
    reconstructed = F.interpolate(reconstructed, size=(rgb_image.shape[1], rgb_image.shape[2]), 
                                  mode='bilinear', align_corners=False)
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # Normalize images
    rgb_np = rgb_image.cpu().numpy().transpose(1, 2, 0)
    rgb_np = (rgb_np - rgb_np.min()) / (rgb_np.max() - rgb_np.min() + 1e-6)
    
    ir_np = ir_image.cpu().numpy()[0]
    ir_np = (ir_np - ir_np.min()) / (ir_np.max() - ir_np.min() + 1e-6)
    
    reconstructed_np = reconstructed[0].cpu().detach().numpy().transpose(1, 2, 0)
    
    # Row 1: Input and reconstruction
    axes[0, 0].imshow(rgb_np)
    axes[0, 0].set_title('RGB Input', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(ir_np, cmap='gray')
    axes[0, 1].set_title('IR Input', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(reconstructed_np)
    axes[0, 2].set_title('Reconstructed Image', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    
    # Blended view
    blended = 0.5 * rgb_np + 0.5 * reconstructed_np
    axes[0, 3].imshow(blended)
    axes[0, 3].set_title('Blended (50/50)', fontsize=12, fontweight='bold')
    axes[0, 3].axis('off')
    
    # Row 2: Analysis
    # Difference
    difference = np.abs(reconstructed_np - rgb_np)
    im1 = axes[1, 0].imshow(difference)
    axes[1, 0].set_title('Absolute Difference', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    plt.colorbar(im1, ax=axes[1, 0])
    
    # Channel-wise difference
    diff_r = np.abs(reconstructed_np[:, :, 0] - rgb_np[:, :, 0])
    im2 = axes[1, 1].imshow(diff_r, cmap='Reds')
    axes[1, 1].set_title('Red Channel Difference', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    plt.colorbar(im2, ax=axes[1, 1])
    
    # Uncertainty
    uncertainty = np.std([reconstructed_np, rgb_np], axis=0).mean(axis=2)
    im3 = axes[1, 2].imshow(uncertainty, cmap='hot')
    axes[1, 2].set_title('Uncertainty Map', fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')
    plt.colorbar(im3, ax=axes[1, 2])
    
    # Latent space stats
    latent_np = latent[0].cpu().detach().numpy()
    axes[1, 3].hist(latent_np, bins=30, edgecolor='black', alpha=0.7)
    axes[1, 3].set_title(f'Latent Distribution (μ={latent_np.mean():.2f}, σ={latent_np.std():.2f})', 
                         fontsize=12, fontweight='bold')
    axes[1, 3].set_xlabel('Value')
    axes[1, 3].set_ylabel('Frequency')
    axes[1, 3].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Reconstruction stages saved to {output_path}")
    
    return fig
