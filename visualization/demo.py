"""Demo script for visualization module."""

import torch
import sys
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

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


def load_real_image_from_huggingface():
    """Load a real image from Hugging Face dataset."""
    try:
        from datasets import load_dataset
        
        print("   Loading Rijksmuseum dataset from Hugging Face...")
        dataset = load_dataset('thers2m/rijksmuseum_painting_dataset', split='train', streaming=True)
        
        # Get first sample
        sample = next(iter(dataset))
        
        # Extract image
        if 'image' in sample:
            image = sample['image']
        elif 'painting' in sample:
            image = sample['painting']
        else:
            # Try first image-like key
            for key in sample.keys():
                if isinstance(sample[key], Image.Image):
                    image = sample[key]
                    break
        
        # Convert to tensor
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        
        image_tensor = transform(image)
        print(f"   ✓ Loaded real image from Hugging Face: {image_tensor.shape}")
        
        return image_tensor
    
    except Exception as e:
        print(f"   ⚠ Could not load from Hugging Face: {e}")
        print("   Using synthetic image instead...")
        return None


def create_synthetic_spectral_channels(image):
    """
    Create synthetic spectral channels from RGB image.
    Simulates IR, UV, and X-ray by applying different filters.
    """
    # Infrared: emphasize red channel
    ir = image[0:1] * 0.7 + image[1:2] * 0.2 + image[2:3] * 0.1
    
    # UV: emphasize blue channel
    uv = image[2:3] * 0.6 + image[0:1] * 0.3 + image[1:2] * 0.1
    
    # X-ray: average all channels with slight noise
    xray = (image[0:1] + image[1:2] + image[2:3]) / 3
    xray = xray + torch.randn_like(xray) * 0.05
    
    return ir, uv, xray


def run_visualization_demo():
    """Run complete visualization demo."""
    
    print("=" * 80)
    print("PAINTING ANALYSIS VISUALIZATION DEMO")
    print("=" * 80)
    
    # Create output directory
    output_dir = Path('visualization_outputs')
    output_dir.mkdir(exist_ok=True)
    
    # Load real image from Hugging Face
    print("\n1. Loading real painting image...")
    image = load_real_image_from_huggingface()
    
    if image is None:
        # Fallback to synthetic
        print("   Generating synthetic image...")
        image = torch.randn(3, 224, 224)
    
    # Normalize image to [0, 1]
    image = (image - image.min()) / (image.max() - image.min() + 1e-6)
    
    # Create synthetic spectral channels
    print("\n2. Creating synthetic spectral channels...")
    ir_image, uv_image, xray_image = create_synthetic_spectral_channels(image)
    print(f"   Image shape: {image.shape}")
    print(f"   IR shape: {ir_image.shape}")
    print(f"   UV shape: {uv_image.shape}")
    print(f"   X-ray shape: {xray_image.shape}")
    
    # Generate model predictions (synthetic)
    print("\n3. Generating model predictions...")
    height, width = image.shape[1], image.shape[2]
    layer_masks = torch.randn(5, height, width)
    pigment_logits = torch.randn(10, height, width)
    print(f"   Layer masks shape: {layer_masks.shape}")
    print(f"   Pigment logits shape: {pigment_logits.shape}")
    
    # 1. Prediction Overlay
    print("\n4. Generating prediction overlay visualization...")
    fig1 = visualize_prediction_overlay(
        image, layer_masks, pigment_logits,
        output_path=output_dir / 'prediction_overlay.png'
    )
    print("   ✓ Saved: prediction_overlay.png")
    
    # 2. Spectral Comparison
    print("\n5. Generating spectral channel comparison...")
    fig2 = visualize_spectral_channels(
        image, ir_image, uv_image, xray_image,
        layer_masks, pigment_logits,
        output_path=output_dir / 'spectral_comparison.png'
    )
    print("   ✓ Saved: spectral_comparison.png")
    
    # 3. Grad-CAM Heatmaps
    print("\n6. Generating Grad-CAM heatmaps...")
    fig3 = visualize_gradcam(
        image, layer_masks, pigment_logits, None,
        output_path=output_dir / 'gradcam_heatmaps.png'
    )
    print("   ✓ Saved: gradcam_heatmaps.png")
    
    # 4. Attention Heatmaps
    print("\n7. Generating attention heatmaps...")
    fig4 = create_attention_heatmap(
        layer_masks, pigment_logits,
        output_path=output_dir / 'attention_heatmaps.png'
    )
    print("   ✓ Saved: attention_heatmaps.png")
    
    # 5. Hidden Image Reconstruction Demo
    print("\n8. Running hidden image reconstruction demo...")
    reconstructor, fig5 = demo_hidden_image_reconstruction(
        output_path=output_dir / 'reconstruction_demo.png'
    )
    print("   ✓ Saved: reconstruction_demo.png")
    
    # 6. Reconstruction Stages
    print("\n9. Generating reconstruction stages visualization...")
    fig6 = visualize_reconstruction_stages(
        image, ir_image, reconstructor,
        output_path=output_dir / 'reconstruction_stages.png'
    )
    print("   ✓ Saved: reconstruction_stages.png")
    
    # 7. Comprehensive Analysis Report
    print("\n10. Generating comprehensive analysis report...")
    fig7 = generate_analysis_report(
        image, layer_masks, pigment_logits,
        output_path=output_dir / 'analysis_report.png'
    )
    print("   ✓ Saved: analysis_report.png")
    
    # 8. Statistics Summary
    print("\n11. Generating statistics summary...")
    fig8, stats = generate_statistics_summary(
        layer_masks, pigment_logits,
        output_path=output_dir / 'statistics_summary.png'
    )
    print("   ✓ Saved: statistics_summary.png")
    
    # Print statistics
    print("\n" + "=" * 80)
    print("STATISTICS SUMMARY")
    print("=" * 80)
    
    print("\nLayer Statistics:")
    for i in range(5):
        coverage = stats[f'layer_{i}_coverage']
        confidence = stats[f'layer_{i}_confidence']
        print(f"  Layer {i}: Coverage={coverage*100:.1f}%, Confidence={confidence:.3f}")
    
    print("\nPigment Statistics (top 5):")
    pigment_names = [
        'Lead White', 'Ultramarine', 'Ochre', 'Vermillion', 'Azurite',
        'Umber', 'Charcoal', 'Lapis Lazuli', 'Cadmium Yellow', 'Titanium White'
    ]
    pigment_coverage = [(i, stats[f'pigment_{i}_coverage']) for i in range(10)]
    pigment_coverage.sort(key=lambda x: x[1], reverse=True)
    
    for idx, (pigment_id, coverage) in enumerate(pigment_coverage[:5]):
        confidence = stats[f'pigment_{pigment_id}_confidence']
        print(f"  {pigment_names[pigment_id]}: Coverage={coverage*100:.1f}%, Confidence={confidence:.3f}")
    
    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)
    print(f"\nAll visualizations saved to: {output_dir.absolute()}")
    print("\nGenerated files:")
    for i, file in enumerate(sorted(output_dir.glob('*.png')), 1):
        print(f"  {i}. {file.name}")
    
    return output_dir


if __name__ == '__main__':
    output_dir = run_visualization_demo()
