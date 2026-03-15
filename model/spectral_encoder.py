import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralAttention(nn.Module):
    """Spectral attention mechanism to learn pigment bands."""
    
    def __init__(self, channels):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // 16)
        self.fc2 = nn.Linear(channels // 16, channels)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x: (B, C, H, W)
        b, c, h, w = x.size()
        
        # Global average pooling
        avg_pool = F.adaptive_avg_pool2d(x, 1).view(b, c)
        
        # Attention weights
        att = self.fc1(avg_pool)
        att = F.relu(att)
        att = self.fc2(att)
        att = self.sigmoid(att).view(b, c, 1, 1)
        
        return x * att


class SpectralEncoder(nn.Module):
    """3D-CNN extracts material features from multispectral image cube."""
    
    def __init__(self, in_channels=100, out_channels=64):
        """
        Args:
            in_channels: Number of spectral bands
            out_channels: Output feature channels
        """
        super().__init__()
        
        # 3D convolution blocks
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.att1 = SpectralAttention(32)
        
        self.conv2 = nn.Conv2d(32, 48, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(48)
        self.att2 = SpectralAttention(48)
        
        self.conv3 = nn.Conv2d(48, out_channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.att3 = SpectralAttention(out_channels)
        
        self.pool = nn.MaxPool2d(2, 2)
    
    def forward(self, x):
        """
        Args:
            x: (B, Bands, H, W) multispectral image
        
        Returns:
            features: (B, 64, H, W) feature cube
        """
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.att1(x)
        
        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.att2(x)
        
        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.att3(x)
        
        return x
