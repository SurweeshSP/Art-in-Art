# Visualization Module

Comprehensive visualization tools for painting analysis and model explainability.

## Overview

The visualization module provides publication-ready visualizations for:
- Prediction overlays and segmentation maps
- Spectral channel comparisons
- Grad-CAM heatmaps for model explainability
- Hidden image reconstruction demonstrations
- Comprehensive analysis reports

## Features

### 1. Prediction Overlays (`prediction_overlay.py`)

Visualize layer segmentation and pigment classification predictions overlaid on original images.

```python
from visualization import visualize_prediction_overlay

fig = visualize_prediction_overlay(
    image, layer_masks, pigment_logits,
    output_path='prediction_overlay.png'
)
```

**Outputs:**
- Original image
- Layer overlay (multiple transparency levels)
- Layer segmentation map
- Pigment classification map
- Confidence maps

### 2. Spectral Channel Comparison (`spectral_comparison.py`)

Compare multispectral imaging modalities with model predictions.

```python
from visualization import visualize_spectral_channels

fig = visualize_spectral_channels(
    rgb_image, ir_image, uv_image, xray_image,
    layer_masks, pigment_logits,
    output_path='spectral_comparison.png'
)
```

**Outputs:**
- RGB, Infrared, UV, X-ray channels
- Layer and pigment predictions
- Confidence maps for each modality

### 3. Grad-CAM Heatmaps (`gradcam.py`)

Explainability visualizations showing model attention regions.

```python
from visualization import visualize_gradcam, create_attention_heatmap

# Grad-CAM heatmaps
fig1 = visualize_gradcam(
    image, layer_masks, pigment_logits, model,
    output_path='gradcam_heatmaps.png'
)

# Attention heatmaps with uncertainty
fig2 = create_attention_heatmap(
    layer_masks, pigment_logits,
    output_path='attention_heatmaps.png'
)
```

**Outputs:**
- Layer and pigment Grad-CAM heatmaps
- Confidence and uncertainty maps
- Combined attention visualization

### 4. Hidden Image Reconstruction (`reconstruction_demo.py`)

Demonstrate the reconstruction pipeline for hidden images.

```python
from visualization import demo_hidden_image_reconstruction, visualize_reconstruction_stages

# Basic reconstruction demo
reconstructor, fig1 = demo_hidden_image_reconstruction(
    output_path='reconstruction_demo.png'
)

# Detailed reconstruction stages
fig2 = visualize_reconstruction_stages(
    image, ir_image, reconstructor,
    output_path='reconstruction_stages.png'
)
```

**Outputs:**
- Input spectral channels
- Latent space representation
- Reconstructed hidden image
- Difference maps
- Uncertainty maps

### 5. Analysis Reports (`analysis_report.py`)

Generate comprehensive analysis reports combining all visualizations.

```python
from visualization import generate_analysis_report, generate_statistics_summary

# Full analysis report
fig1 = generate_analysis_report(
    image, layer_masks, pigment_logits,
    output_path='analysis_report.png'
)

# Statistics summary
fig2, stats = generate_statistics_summary(
    layer_masks, pigment_logits,
    output_path='statistics_summary.png'
)
```

**Outputs:**
- 4×4 grid with all key visualizations
- Layer and pigment coverage statistics
- Confidence distributions
- Reconstruction results (if available)

## Running the Demo

The demo script generates all visualizations using real images from Hugging Face:

```bash
python visualization/demo.py
```

**Features:**
- Loads real paintings from Rijksmuseum dataset
- Creates synthetic spectral channels (IR, UV, X-ray)
- Generates all visualization types
- Saves outputs to `visualization_outputs/`
- Prints statistics summary

**Output:**
```
================================================================================
PAINTING ANALYSIS VISUALIZATION DEMO
================================================================================

1. Loading real painting image...
   ✓ Loaded real image from Hugging Face: torch.Size([3, 224, 224])

2. Creating synthetic spectral channels...
   Image shape: torch.Size([3, 224, 224])
   ...

Generated files:
  1. analysis_report.png
  2. attention_heatmaps.png
  3. gradcam_heatmaps.png
  4. prediction_overlay.png
  5. reconstruction_demo.png
  6. reconstruction_stages.png
  7. spectral_comparison.png
  8. statistics_summary.png
```

## Module Structure

```
visualization/
├── __init__.py                    # Module exports
├── prediction_overlay.py          # Overlay visualizations
├── spectral_comparison.py         # Spectral channel analysis
├── gradcam.py                     # Explainability heatmaps
├── reconstruction_demo.py         # Hidden image reconstruction
├── analysis_report.py             # Comprehensive reports
├── demo.py                        # Demo script
└── README.md                      # This file
```

## Data Requirements

### Input Tensors

All visualization functions expect PyTorch tensors:

- **image**: `(3, H, W)` RGB image, values in [0, 1]
- **layer_masks**: `(5, H, W)` layer segmentation logits
- **pigment_logits**: `(10, H, W)` pigment classification logits
- **ir_image**: `(1, H, W)` infrared channel
- **uv_image**: `(1, H, W)` UV fluorescence channel
- **xray_image**: `(1, H, W)` X-ray radiography channel

### Data Sources

The demo loads real images from:
- **Rijksmuseum Technical Imaging Dataset** (Hugging Face)
- Automatically downloads on first run
- Requires internet connection

## Output Formats

All visualizations are saved as high-resolution PNG files:
- **DPI**: 150 (publication quality)
- **Format**: PNG with tight bounding box
- **Size**: Varies by visualization (typically 2-4 MB)

## Customization

### Modify Colors

```python
# Change colormap for layer segmentation
im = ax.imshow(layer_pred, cmap='viridis')  # Instead of 'tab10'
```

### Adjust Transparency

```python
# Change overlay transparency
overlaid = (1 - alpha) * image_np + alpha * layer_colored
# alpha=0.3 for light overlay, alpha=0.7 for strong overlay
```

### Save to Different Format

```python
# Save as PDF instead of PNG
plt.savefig('output.pdf', dpi=300, bbox_inches='tight')
```

## Performance

- **Prediction Overlay**: ~2 seconds
- **Spectral Comparison**: ~3 seconds
- **Grad-CAM Heatmaps**: ~2 seconds
- **Reconstruction Demo**: ~5 seconds
- **Analysis Report**: ~4 seconds
- **Total Demo**: ~20 seconds

## Dependencies

```
torch
torchvision
matplotlib
numpy
PIL
datasets (for Hugging Face)
```

## Citation

If you use these visualizations in research, please cite:

```bibtex
@software{painting_analysis_2026,
  title={Painting in a Painting: Visualization Module},
  author={Your Name},
  year={2026},
  url={https://github.com/your-repo}
}
```

## License

MIT License - See LICENSE file for details

## Support

For issues or questions:
1. Check the demo script for usage examples
2. Review docstrings in each module
3. Refer to the main README.md for project context
