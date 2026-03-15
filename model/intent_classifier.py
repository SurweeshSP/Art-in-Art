import torch
import torch.nn as nn
import torch.nn.functional as F


class IntentClassifier(nn.Module):
    """Analyzes reconstruction trajectory to classify modification motivation."""
    
    def __init__(self, feature_dim=256, num_intents=5):
        """
        Args:
            feature_dim: Dimension of input features
            num_intents: Number of intent classes
                0: Artistic Revision
                1: Damage Repair
                2: Canvas Reuse
                3: Censorship/Concealment
                4: Pentimento (artist's change of mind)
        """
        super().__init__()
        self.num_intents = num_intents
        
        # Feature extraction from reconstruction trajectory
        self.trajectory_encoder = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        
        # Spectral analysis branch
        self.spectral_analyzer = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
        )
        
        # Layer difference analysis
        self.layer_analyzer = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
        )
        
        # Fusion and classification
        self.fusion = nn.Sequential(
            nn.Linear(256 + 128 + 128, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        
        # Intent classification head
        self.intent_head = nn.Linear(256, num_intents)
        
        # Confidence estimation
        self.confidence_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, trajectory_features, spectral_features, layer_features):
        """
        Args:
            trajectory_features: (B, feature_dim) reconstruction trajectory
            spectral_features: (B, feature_dim) spectral signature changes
            layer_features: (B, feature_dim) layer difference features
        
        Returns:
            intent_logits: (B, num_intents) classification logits
            confidence: (B, 1) confidence scores
            intent_probs: (B, num_intents) probability distribution
        """
        # Encode each branch
        traj_encoded = self.trajectory_encoder(trajectory_features)
        spec_encoded = self.spectral_analyzer(spectral_features)
        layer_encoded = self.layer_analyzer(layer_features)
        
        # Fuse features
        fused = torch.cat([traj_encoded, spec_encoded, layer_encoded], dim=1)
        fused_features = self.fusion(fused)
        
        # Classification
        intent_logits = self.intent_head(fused_features)
        confidence = self.confidence_head(fused_features)
        intent_probs = F.softmax(intent_logits, dim=1)
        
        return intent_logits, confidence, intent_probs


class IntentAnalyzer(nn.Module):
    """Analyzes reconstruction trajectory and extracts intent features."""
    
    def __init__(self, num_steps=10, feature_dim=256):
        """
        Args:
            num_steps: Number of reconstruction steps to analyze
            feature_dim: Output feature dimension
        """
        super().__init__()
        self.num_steps = num_steps
        
        # Temporal encoder for reconstruction steps
        self.temporal_encoder = nn.LSTM(
            input_size=3,  # RGB channels
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        # Feature projection
        self.feature_projection = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim)
        )
    
    def forward(self, reconstruction_steps):
        """
        Args:
            reconstruction_steps: (B, num_steps, 3, H, W) reconstruction trajectory
        
        Returns:
            trajectory_features: (B, feature_dim) encoded trajectory
        """
        b, steps, c, h, w = reconstruction_steps.size()
        
        # Flatten spatial dimensions
        steps_flat = reconstruction_steps.view(b, steps, -1)
        
        # Encode temporal sequence
        _, (hidden, _) = self.temporal_encoder(steps_flat)
        trajectory_features = hidden[-1]  # Last hidden state
        
        # Project to feature dimension
        trajectory_features = self.feature_projection(trajectory_features)
        
        return trajectory_features


class IntentLoss(nn.Module):
    """Custom loss for intent classification with confidence weighting."""
    
    def __init__(self, num_intents=5):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.num_intents = num_intents
    
    def forward(self, intent_logits, confidence, targets):
        """
        Args:
            intent_logits: (B, num_intents) classification logits
            confidence: (B, 1) confidence scores
            targets: (B,) ground truth intent labels
        
        Returns:
            loss: Weighted classification loss
        """
        ce = self.ce_loss(intent_logits, targets)
        
        # Weight loss by confidence
        weighted_loss = ce * confidence.mean()
        
        return weighted_loss
