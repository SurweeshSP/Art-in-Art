"""Visualization module for painting analysis."""

from .prediction_overlay import create_prediction_overlay, visualize_prediction_overlay
from .spectral_comparison import visualize_spectral_channels, create_spectral_difference_map
from .gradcam import GradCAM, visualize_gradcam, create_attention_heatmap
from .reconstruction_demo import (
    SimpleHiddenImageReconstructor, 
    demo_hidden_image_reconstruction,
    visualize_reconstruction_stages
)
from .analysis_report import generate_analysis_report, generate_statistics_summary

__all__ = [
    'create_prediction_overlay',
    'visualize_prediction_overlay',
    'visualize_spectral_channels',
    'create_spectral_difference_map',
    'GradCAM',
    'visualize_gradcam',
    'create_attention_heatmap',
    'SimpleHiddenImageReconstructor',
    'demo_hidden_image_reconstruction',
    'visualize_reconstruction_stages',
    'generate_analysis_report',
    'generate_statistics_summary',
]
