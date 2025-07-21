"""
Millimap: Spatial omics viewer package.

A lightweight viewer for spatial omics AnnData objects (.h5ad) with 3D visualization,
interactive filtering, and analysis tools.
"""

from .viewer import MillimapViewer
from .main import main

__version__ = "1.0.0"
__author__ = "Milliomics Team"

__all__ = ["MillimapViewer", "main"] 