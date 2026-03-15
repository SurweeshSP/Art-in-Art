import torch
import torch.nn as nn
import torch.optim as optim
import logging
from tqdm import tqdm
from pathlib import Path

from data.ingestion import get_dataloaders
from model import (
    SpectralEncoder,
    StratigraphicDecoder,
    PalimpsestReconstructor,
    IntentClassifier,
    IntentLoss
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PalimpsestPipeline(nn.Module):
    """Complete pipeline: Encoder → Decoder → Reconstructor → Intent Classifier."""
    
    def __init__(self, num_spectral_bands=3, num_layers=5, num_pigments=10, num_intents=5):
        super().__init__()
        # Adapt encoder to accept RGB (3 channels) and expand to spectral features
        self.channel_expansion = nn.Conv2d(num_spectral_bands, 64, kernel_size=1)
        self.encoder = SpectralEncoder(in_channels=64, out_channels=64)
        self.decoder = StratigraphicDecoder(in_channels=64, num_layers=num_layers, num_pigments=num_pigments)
        self.reconstructor = PalimpsestReconstructor(input_depth=32, num_channels=128)
        self.intent_classifier = IntentClassifier(feature_dim=256, num_intents=num_intents)
    
    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) RGB image
        
        Returns:
            features: encoder output
            layer_masks: segmentation masks
            pigment_logits: pigment classification
            reconstructed: hidden image
        """
        # Expand RGB to spectral features
        x = self.channel_expansion(x)
        
        # Stage 1: Spectral encoding
        features = self.encoder(x)
        
        # Stage 2: Stratigraphic decoding
        layer_masks, pigment_logits = self.decoder(features)
        
        # Stage 3: Reconstruction
        batch_size = x.size(0)
        random_input = torch.randn(batch_size, 32, x.size(2), x.size(3), device=x.device)
        reconstructed = self.reconstructor(random_input)
        
        return features, layer_masks, pigment_logits, reconstructed


def train_epoch(model, train_loader, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)
        
        # Forward pass
        features, layer_masks, pigment_logits, reconstructed = model(images)
        
        # Simple reconstruction loss (MSE between input and reconstruction)
        # In practice, you'd have ground truth hidden images
        loss = nn.functional.mse_loss(reconstructed, images[:, :3, :, :])
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(train_loader)
    logger.info(f"Epoch {epoch} - Avg Loss: {avg_loss:.4f}")
    return avg_loss


def validate(model, val_loader, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            images = batch['image'].to(device)
            
            features, layer_masks, pigment_logits, reconstructed = model(images)
            loss = nn.functional.mse_loss(reconstructed, images[:, :3, :, :])
            total_loss += loss.item()
    
    avg_loss = total_loss / len(val_loader)
    logger.info(f"Validation Loss: {avg_loss:.4f}")
    return avg_loss


def main():
    # Configuration
    config = {
        'batch_size': 2,  # Reduced from 8 to fit in 4GB GPU
        'num_epochs': 10,
        'learning_rate': 1e-3,
        'num_workers': 0,  # Set to 0 for Windows
        'img_size': 224,  # Reduced from 256 to save memory
        'num_spectral_bands': 3,  # RGB channels
        'num_layers': 5,
        'num_pigments': 10,
        'num_intents': 5,
    }
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create dataloaders
    logger.info("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        img_size=config['img_size']
    )
    
    # Create model
    logger.info("Creating model...")
    model = PalimpsestPipeline(
        num_spectral_bands=config['num_spectral_bands'],
        num_layers=config['num_layers'],
        num_pigments=config['num_pigments'],
        num_intents=config['num_intents']
    ).to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    # Training loop
    logger.info("Starting training...")
    best_val_loss = float('inf')
    checkpoint_dir = Path('checkpoints')
    checkpoint_dir.mkdir(exist_ok=True)
    
    for epoch in range(1, config['num_epochs'] + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)
        val_loss = validate(model, val_loader, device)
        scheduler.step()
        
        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = checkpoint_dir / f'best_model.pt'
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"Saved best model to {checkpoint_path}")
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
