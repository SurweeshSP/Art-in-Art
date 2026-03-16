# Painting in a Painting — Hidden Image Detection with AI

Exploring hidden artwork beneath paintings using artificial intelligence and multispectral imaging.

This project develops a deep learning system for analyzing paintings using multispectral images to detect hidden artwork, identify painting layers, and reconstruct concealed compositions. The system integrates computer vision, spectral analysis, and explainable AI to support art historians and conservation experts.

**GSoC Project**: Art-in-Art Detection and Analysis  
**Mentors**: Emanuele Usai (University of Alabama), Sergei Gleyzer (University of Alabama)

---

## Project Information

### Project Title
Painting in a Painting – AI-Based Hidden Image Detection and Reconstruction

### Project Motivation

Artists historically reused canvases or modified paintings over time. As a result, many artworks contain hidden sketches, abandoned compositions, or repainted elements beneath the visible surface.

Modern imaging techniques such as:
- Infrared reflectography
- Ultraviolet fluorescence
- X-ray radiography
- Multispectral imaging

allow researchers to observe deeper layers of paintings. However, analyzing these images manually is complex and requires expert knowledge.

This project aims to develop an AI-powered system that automatically analyzes multispectral scans to detect hidden artwork and understand the structural composition of paintings.

### Core Objectives

The system aims to solve three main problems:

1. **Painting Property Analysis**
   - Identify structural and material properties of paintings
   - Pigment distribution
   - Restoration regions
   - Surface damage
   - Underdrawings

2. **Hidden Image Detection**
   - Automatically determine whether a painting contains hidden imagery beneath the visible layer

3. **Hidden Image Reconstruction**
   - Estimate or reconstruct the underlying hidden painting using deep learning models trained on multispectral data

---

## System Architecture

The project implements a multi-stage deep learning pipeline:

```
Multispectral Painting Data (RGB + Infrared + UV + X-ray)
          ↓
Canvas Encoder — Multispectral feature extraction
          ↓
Stratigraphic Decoder — Layer segmentation and pigment analysis
          ↓
Palimpsest Detector — Hidden image detection
          ↓
Palimpsest Reconstructor — Hidden painting reconstruction
          ↓
Explainability Module — Grad-CAM visualizations
```

### Pipeline Components

#### 1. Canvas Encoder

The Canvas Encoder extracts multispectral features from input images.

**Architecture:**
- ResNet-based convolutional encoder with spectral attention
- Input: `[batch, channels, height, width]` where `channels = RGB + IR + UV + X-ray`
- Output: Feature cube `[batch, 64, H/4, W/4]`

**Purpose:**
- Fuse spectral information
- Learn painting textures and material patterns
- Capture underlying structures

#### 2. Stratigraphic Decoder

The Stratigraphic Decoder analyzes painting layers using a U-Net architecture enhanced with transformer-based attention mechanisms.

**Architecture:**

The decoder follows a symmetric encoder-decoder structure:

```
Input Feature Cube (B, 64, H, W)
          ↓
Encoder Block 1: Conv2d(64→64) + MaxPool2d
          ↓
Encoder Block 2: Conv2d(64→128) + MaxPool2d
          ↓
Bottleneck: Conv2d(128→256) + TransformerBlock(num_heads=4)
          ↓
Decoder Block 2: ConvTranspose2d(256→128) + Skip Connection + Conv2d(256→128)
          ↓
Decoder Block 1: ConvTranspose2d(128→64) + Skip Connection + Conv2d(128→64)
          ↓
Output Heads:
  ├─ Layer Segmentation: Conv2d(64→num_layers)
  └─ Pigment Classification: Conv2d(64→num_pigments)
```

**Key Components:**

1. **U-Net Encoder**
   - Progressive downsampling with max pooling
   - Captures multi-scale features
   - Preserves spatial information through skip connections

2. **Transformer Bottleneck**
   - Multi-head self-attention (4 heads, 256 channels)
   - Captures long-range dependencies between painting regions
   - Layer normalization and feed-forward networks
   - Helps identify relationships between distant layer regions

3. **U-Net Decoder**
   - Progressive upsampling with transposed convolutions
   - Skip connections from encoder preserve fine details
   - Concatenation of encoder and decoder features
   - Restores spatial resolution for pixel-level predictions

4. **Output Heads**
   - **Layer Segmentation Head**: Produces `(B, num_layers, H, W)` masks
     - Each channel represents probability of a specific layer
     - Softmax applied during inference for layer assignment
   - **Pigment Classification Head**: Produces `(B, num_pigments, H, W)` logits
     - Identifies pigment types at each pixel location
     - Supports multi-pigment regions (overlapping predictions)

**Outputs:**

The decoder produces three main outputs:

1. **Layer Segmentation Masks** `(B, num_layers, H, W)`
   - Binary or probabilistic masks for each painting layer
   - Identifies which pixels belong to which layer
   - Enables layer-by-layer analysis

2. **Pigment Classification Maps** `(B, num_pigments, H, W)`
   - Pixel-wise pigment identification
   - Supports 10+ pigment classes (ochre, ultramarine, lead white, etc.)
   - Helps conservators understand material composition

3. **Restoration Detection** (implicit in layer masks)
   - Anomalous regions indicate restoration or overpainting
   - Detected through layer discontinuities and pigment inconsistencies

**Example Output Visualization:**

```
Original RGB Image          Infrared Reflectography
    ↓                              ↓
    └──────────────┬───────────────┘
                   ↓
        Stratigraphic Decoder
                   ↓
    ┌──────────────┬──────────────┐
    ↓              ↓              ↓
Layer Masks    Pigment Map    Damage Map
(5 layers)     (10 pigments)   (restoration)
```

**Purpose:**

- **Understand the physical structure** of the painting through layer decomposition
- **Identify material composition** by detecting pigment distributions
- **Detect conservation interventions** through anomaly detection in layer patterns
- **Support art historians** with quantitative layer analysis
- **Enable restoration planning** by identifying damaged or altered regions

**Training Details:**

- **Loss Function**: Combination of cross-entropy (layer segmentation) and focal loss (pigment classification)
- **Optimization**: Adam optimizer with learning rate scheduling
- **Regularization**: Batch normalization, dropout in transformer blocks
- **Metrics**: IoU (Intersection over Union) for segmentation, F1-score for pigment classification

**Computational Efficiency:**

- Transformer attention operates on flattened spatial dimensions: `(B, H*W, C)`
- Reduces memory complexity compared to full 2D attention
- 4 attention heads balance expressiveness and memory usage
- Skip connections enable efficient gradient flow during backpropagation

**Processing Output for Visualization:**

**Detailed Output Analysis:**

**Integration with Full Pipeline:**

**Expected Console Output:**

---

## Visualization & Explainability

### 1. Prediction Overlays

Visualize model predictions directly on the original image:

**Output:** Side-by-side comparison showing original image, semi-transparent overlay, and pure segmentation map.

---

### 2. Spectral Channel Comparison

Compare different spectral modalities to understand layer detection:

**Output:** 2×4 grid showing all spectral modalities alongside model predictions and confidence maps.

---

### 3. Grad-CAM Heatmaps (Explainability)

Visualize which regions the model focuses on for layer detection:

**Output:** 2×3 grid showing original image, predictions, and Grad-CAM heatmaps overlaid on the image.

---

### 4. Hidden Image Reconstruction Demo

Demonstrate the reconstruction pipeline with a simple autoencoder approach:

**Output:** 2×3 grid showing:
- Input RGB and infrared channels
- Latent space representation
- Reconstructed hidden image
- Difference map highlighting hidden content
- Uncertainty/confidence map

---

### 5. Complete Visualization Pipeline

Combine all visualizations into a comprehensive analysis report:

---

## Visualization Output Examples

These visualizations demonstrate:

✓ **Scientific Rigor**: Spectral channel comparisons show multi-modal analysis  
✓ **Explainability**: Grad-CAM heatmaps reveal model decision-making  
✓ **Reconstruction Quality**: Side-by-side comparisons of input vs. reconstructed images  
✓ **Confidence Metrics**: Uncertainty maps show model reliability  
✓ **Layer Analysis**: Segmentation masks reveal painting structure  
✓ **Pigment Detection**: Classification maps identify material composition  

These outputs are publication-ready and suitable for research papers, museum exhibitions, and conservation reports.

#### 3. Palimpsest Detector

This module determines whether a painting contains hidden artwork beneath the visible surface.

**Architecture:**
- Multispectral classifier using features from the Canvas Encoder
- Output: `hidden_image_probability`, `confidence_score`

**Example Output:**
```
Hidden Image: True
Confidence: 0.87
```

#### 4. Palimpsest Reconstructor

If hidden artwork is detected, this stage attempts to reconstruct it.

**Architecture:**
- Generative models such as:
  - Pix2Pix GAN
  - Diffusion-based image reconstruction

**Input:** Infrared / X-ray spectral channels  
**Output:** Estimated hidden painting

This stage provides a visual hypothesis of the concealed composition.

#### 5. Explainability Module

Explainable AI helps art historians interpret the model's decisions.

**Techniques used:**
- Grad-CAM visualization
- Output: Attention heatmap highlighting hidden structures

This allows experts to verify whether the model focuses on meaningful regions.

---

## Dataset Strategy

Multispectral art datasets are relatively small, so the training pipeline combines multiple sources.

### 1. Multispectral Painting Dataset

**Primary dataset:** Rijksmuseum Technical Imaging Dataset

**Contains:**
- RGB images
- Infrared reflectography
- X-ray radiographs
- High-resolution artwork scans

### 2. Cultural Heritage Imaging Dataset

**Used for:**
- Multispectral analysis
- Pigment detection
- Restoration studies

### 3. WikiArt Dataset

**Used for pretraining**
- 80,000+ paintings
- Multiple art styles
- Artist metadata

**Purpose:** Learn general painting features

### 4. Met Museum Open Access Dataset

**Provides:** 470,000+ artwork images

**Used for:**
- Transfer learning
- Feature pretraining

### 5. Synthetic Hidden Image Dataset

To improve training, synthetic examples will be generated.

**Method:**
- Overlay two paintings
- Example: `visible_painting + hidden_painting → simulated multispectral image`

This helps train models to detect hidden layers.

### Dataset Structure

```
dataset/
  painting_001/
    rgb.png
    infrared.png
    uv.png
    xray.png
  painting_002/
    rgb.png
    infrared.png
    uv.png
    xray.png
```

**Model input tensor:** `[channels, height, width]` where `channels = 4`

---

## Training Strategy

Training occurs in multiple stages.

### Stage 1 — Feature Pretraining

**Datasets:** WikiArt, Met Museum  
**Goal:** Learn painting textures and style features

### Stage 2 — Multispectral Learning

**Dataset:** Rijksmuseum technical imaging  
**Goal:** Learn spectral relationships between layers

### Stage 3 — Hidden Image Detection

**Goal:** Train classifier to detect hidden paintings  
**Target metric:** >85% detection accuracy

### Stage 4 — Hidden Image Reconstruction

**Goal:** Train generative model to reconstruct hidden images  
**Evaluation metrics:** SSIM, PSNR

---

## Evaluation Metrics

### Hidden Image Detection
- Accuracy
- Precision
- Recall
- F1 score
- ROC-AUC

### Layer Segmentation
- IoU (Intersection over Union)
- Dice score

### Image Reconstruction
- SSIM (Structural Similarity Index)
- PSNR (Peak Signal-to-Noise Ratio)

---

## Technology Stack

### Programming
- Python

### Deep Learning
- PyTorch
- TensorFlow

### Computer Vision
- OpenCV
- scikit-image

### Data Processing
- NumPy
- Pandas

### Visualization
- Matplotlib
- Plotly

---

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Test Data Loading

```bash
python data/ingestion.py
```

This will load a sample batch and print shapes.

### Train the Model

```bash
python train.py
```

**What happens:**
- Loads Rijksmuseum dataset (train/val/test splits)
- Creates the multi-stage pipeline
- Trains for 10 epochs
- Saves best model to `checkpoints/best_model.pt`

### Expected Output

```
Using device: cuda
Loading data...
Loaded 3180 samples
Train: 398 batches | Val: 50 batches | Test: 50 batches
Creating model...
Starting training...
Epoch 1 - Avg Loss: 0.2345
Validation Loss: 0.1234
...
Training complete!
```

---

## Project Structure

```
├── data/
│   ├── __init__.py
│   └── ingestion.py                    # Dataset loading & preprocessing
├── model/
│   ├── __init__.py
│   ├── spectral_encoder.py             # Canvas Encoder: Feature extraction
│   ├── stratigraphic_decoder.py        # Stratigraphic Decoder: Layer segmentation
│   ├── palimpsest_reconstructor.py     # Palimpsest Reconstructor: Hidden image reconstruction
│   └── intent_classifier.py            # Intent Classifier: Modification analysis
├── checkpoints/                        # Saved model checkpoints
├── train.py                            # Training script
├── requirements.txt                    # Dependencies
└── README.md                           # This file
```

---

## Configuration

Edit `train.py` config dict to customize:

```python
config = {
    'batch_size': 2,           # Batch size (reduced for 4GB GPU)
    'num_epochs': 10,          # Number of training epochs
    'learning_rate': 1e-3,     # Adam learning rate
    'img_size': 224,           # Image resolution (224x224)
    'num_spectral_bands': 3,   # RGB channels (expandable to 4+ for IR/UV/X-ray)
    'num_layers': 5,           # Number of painting layers to segment
    'num_pigments': 10,        # Number of pigment classes
    'num_intents': 5,          # Number of modification intent classes
}
```

---

## GPU Usage

The script automatically uses GPU if available. To force CPU:

```python
# In train.py, change:
device = torch.device('cpu')
```

---

## Troubleshooting

### Out of Memory (OOM)?

- Reduce `batch_size` in config (try 1-2)
- Reduce `img_size` (try 192 or 160)
- Reduce number of transformer heads in `stratigraphic_decoder.py`

### Dataset not found?

- First run downloads from Hugging Face
- Requires internet connection
- Takes ~5-10 minutes first time

### Slow training?

- Reduce `num_workers` if CPU bottleneck
- Use GPU (CUDA)
- Reduce `img_size`

---

## Expected Results

By the end of the project, the system should produce:

### Multispectral Painting Dataset
- Curated dataset with aligned spectral images
- Target size: 1500+ paintings

### AI Models
Models capable of:
- Painting property analysis
- Hidden image detection
- Hidden image reconstruction

### Visualization Interface
Tool for exploring results:
- Original painting
- Spectral scans
- AI detected hidden regions
- Attention heatmaps
- Reconstructed hidden painting

---

## Project Impact

This research can support:
- Art conservation
- Historical painting analysis
- Digital heritage preservation
- Automated museum research tools

The system could help art historians discover hidden artwork faster and more efficiently.

---

## Future Extensions

Possible future directions include:
- Hyperspectral transformer models
- 3D paint layer reconstruction
- Pigment chemistry prediction
- Automated restoration simulation

---

## References

- Rijksmuseum Technical Imaging Dataset
- WikiArt Dataset
- Met Museum Open Access Collection
- PyTorch Documentation
- OpenCV Documentation

---

**Last Updated:** March 2026  
**Status:** Active Development
