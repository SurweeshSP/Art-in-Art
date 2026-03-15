import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerBlock(nn.Module):
    """Transformer block for modeling layer occlusion."""
    
    def __init__(self, channels, num_heads=4):  # Reduced from 8 to 4
        super().__init__()
        self.attention = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.ReLU(),
            nn.Linear(channels * 4, channels)
        )
    
    def forward(self, x):
        # x: (B, C, H, W)
        b, c, h, w = x.size()
        
        # Reshape for attention
        x_flat = x.view(b, c, -1).transpose(1, 2)  # (B, H*W, C)
        
        # Self-attention
        attn_out, _ = self.attention(x_flat, x_flat, x_flat)
        x_flat = self.norm1(x_flat + attn_out)
        
        # MLP
        mlp_out = self.mlp(x_flat)
        x_flat = self.norm2(x_flat + mlp_out)
        
        # Reshape back
        x = x_flat.transpose(1, 2).view(b, c, h, w)
        return x


class UNetBlock(nn.Module):
    """U-Net block for segmentation."""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x


class StratigraphicDecoder(nn.Module):
    """U-Net segments layers and classifies pigments per layer."""
    
    def __init__(self, in_channels=64, num_layers=5, num_pigments=10):
        """
        Args:
            in_channels: Input feature channels
            num_layers: Number of painting layers to segment
            num_pigments: Number of pigment classes
        """
        super().__init__()
        self.num_layers = num_layers
        self.num_pigments = num_pigments
        
        # Encoder
        self.enc1 = UNetBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.enc2 = UNetBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Bottleneck with Transformer
        self.bottleneck = UNetBlock(128, 256)
        self.transformer = TransformerBlock(256, num_heads=4)  # Reduced from 8 to 4
        
        # Decoder
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = UNetBlock(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = UNetBlock(128, 64)
        
        # Output heads
        self.layer_segmentation = nn.Conv2d(64, num_layers, 1)
        self.pigment_classification = nn.Conv2d(64, num_pigments, 1)
    
    def forward(self, x):
        """
        Args:
            x: (B, 64, H, W) feature cube from SpectralEncoder
        
        Returns:
            layer_masks: (B, num_layers, H, W)
            pigment_logits: (B, num_pigments, H, W)
        """
        # Encoder
        enc1 = self.enc1(x)
        x = self.pool1(enc1)
        
        enc2 = self.enc2(x)
        x = self.pool2(enc2)
        
        # Bottleneck
        x = self.bottleneck(x)
        x = self.transformer(x)
        
        # Decoder
        x = self.upconv2(x)
        x = torch.cat([x, enc2], dim=1)
        x = self.dec2(x)
        
        x = self.upconv1(x)
        x = torch.cat([x, enc1], dim=1)
        x = self.dec1(x)
        
        # Output
        layer_masks = self.layer_segmentation(x)
        pigment_logits = self.pigment_classification(x)
        
        return layer_masks, pigment_logits
