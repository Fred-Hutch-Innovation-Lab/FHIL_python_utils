"""
Utilities for single-cell analyses
"""

from .nmad_filter import nmad_filter
from .ridge_plot import ridge_plot
from .daniel_processing import daniel_processing
from .multi_object_embedding_scatter import multi_object_embedding_scatter

__all__ = [
    'nmad_filter',
    'ridge_plot',
    'daniel_processing',
    'multi_object_embedding_scatter'
]
