import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepImagePrior(nn.Module):
    """Deep Image Prior for reconstructing hidden images."""
    
    def __init__(self, input_depth=32, output_depth=3, num_channels=128):
        """
        Args:
            input_depth: Depth of random input
            output_depth: Output channels (3 for RGB)
            num_channels: Base number of channels
        """
        super().__init__()
        self.input_depth = input_depth
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(input_depth, num_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(num_channels, num_channels * 2, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(num_channels * 2, num_channels * 4, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.upconv2 = nn.ConvTranspose2d(num_channels * 4, num_channels * 2, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(num_channels * 2, num_channels * 2, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.upconv1 = nn.ConvTranspose2d(num_channels * 2, num_channels, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Output
        self.output = nn.Conv2d(num_channels, output_depth, 1)
    
    def forward(self, x):
        """
        Args:
            x: (B, input_depth, H, W) random input
        
        Returns:
            reconstructed: (B, 3, H, W) RGB image
        """
        enc1 = self.enc1(x)
        x = self.pool1(enc1)
        
        enc2 = self.enc2(x)
        x = self.pool2(enc2)
        
        x = self.bottleneck(x)
        
        x = self.upconv2(x)
        x = self.dec2(x)
        
        x = self.upconv1(x)
        x = self.dec1(x)
        
        x = self.output(x)
        x = torch.sigmoid(x)  # Normalize to [0, 1]
        
        return x


class StyleCoherence(nn.Module):
    """Maintains artist hand consistency in reconstruction."""
    
    def __init__(self, channels=64):
        super().__init__()
        self.gram_conv = nn.Conv2d(channels, channels, 3, padding=1)
    
    def gram_matrix(self, x):
        """Compute Gram matrix for style."""
        b, c, h, w = x.size()
        x = x.view(b, c, -1)
        gram = torch.bmm(x, x.transpose(1, 2)) / (c * h * w)
        return gram
    
    def forward(self, style_features, content_features):
        """
        Args:
            style_features: Features from visible layer
            content_features: Features from reconstruction
        
        Returns:
            style_loss: Gram matrix difference
        """
        style_gram = self.gram_matrix(style_features)
        content_gram = self.gram_matrix(content_features)
        
        loss = F.mse_loss(style_gram, content_gram)
        return loss


class PalimpsestReconstructor(nn.Module):
    """Reconstructs hidden image with spectral consistency."""
    
    def __init__(self, input_depth=32, num_channels=128):
        super().__init__()
        self.dip = DeepImagePrior(input_depth, output_depth=3, num_channels=num_channels)
        self.style_coherence = StyleCoherence(channels=num_channels)
    
    def forward(self, x, style_features=None):
        """
        Args:
            x: (B, input_depth, H, W) random input
            style_features: Optional style reference features
        
        Returns:
            reconstructed: (B, 3, H, W) hidden image
        """
        reconstructed = self.dip(x)
        return reconstructed
    
    def compute_style_loss(self, style_features, content_features):
        """Compute style coherence loss."""
        return self.style_coherence(style_features, content_features)
