# Quick Start Guide

## Run the Demo

```bash
python visualization/demo.py
```

This generates 8 publication-ready visualizations using real paintings from Hugging Face.

## Basic Usage

```python
from visualization import visualize_prediction_overlay
import torch

# Your data
image = torch.randn(3, 224, 224)
layer_masks = torch.randn(5, 224, 224)
pigment_logits = torch.randn(10, 224, 224)

# Create visualization
fig = visualize_prediction_overlay(
    image, layer_masks, pigment_logits,
    output_path='output.png'
)
```

## All Visualizations

```python
from visualization import (
    visualize_prediction_overlay,
    visualize_spectral_channels,
    visualize_gradcam,
    create_attention_heatmap,
    demo_hidden_image_reconstruction,
    visualize_reconstruction_stages,
    generate_analysis_report,
    generate_statistics_summary,
)

# 1. Prediction overlays
visualize_prediction_overlay(image, layer_masks, pigment_logits)

# 2. Spectral comparison
visualize_spectral_channels(rgb, ir, uv, xray, layer_masks, pigment_logits)

# 3. Grad-CAM heatmaps
visualize_gradcam(image, layer_masks, pigment_logits, model)

# 4. Attention heatmaps
create_attention_heatmap(layer_masks, pigment_logits)

# 5. Reconstruction demo
reconstructor, fig = demo_hidden_image_reconstruction()

# 6. Reconstruction stages
visualize_reconstruction_stages(image, ir_image, reconstructor)

# 7. Analysis report
generate_analysis_report(image, layer_masks, pigment_logits)

# 8. Statistics
fig, stats = generate_statistics_summary(layer_masks, pigment_logits)
```

## Output Directory

All visualizations save to `visualization_outputs/`:
- `prediction_overlay.png`
- `spectral_comparison.png`
- `gradcam_heatmaps.png`
- `attention_heatmaps.png`
- `reconstruction_demo.png`
- `reconstruction_stages.png`
- `analysis_report.png`
- `statistics_summary.png`

## Load Real Images

```python
from visualization.demo import load_real_image_from_huggingface

image = load_real_image_from_huggingface()
```

## Documentation

- Full docs: `visualization/README.md`
- Implementation summary: `VISUALIZATION_SUMMARY.md`
