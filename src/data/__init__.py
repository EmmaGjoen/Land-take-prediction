"""
Data loading and preprocessing utilities for land-take prediction.

This module provides datasets, transforms, and split utilities for
training segmentation models on HABLOSS satellite imagery.
"""

from src.data.sentinel_dataset import SentinelDataset
from src.data.tessera_dataset import TesseraDataset
from src.data.alphaearth_dataset import AlphaEarthDataset
from src.data.splits import get_splits
from src.data.file_helpers import get_ref_ids_from_directory, find_file_by_prefix
from src.data.transform import (
    compute_normalization_stats,
    Normalize,
    NormalizeBy,
    RandomCropTS,
    CenterCropTS,
    ComposeTS,
)

__all__ = [
    # Datasets
    "SentinelDataset",
    "TesseraDataset",
    "AlphaEarthDataset",
    # Splits
    "get_splits",
    # File helpers
    "get_ref_ids_from_directory",
    "find_file_by_prefix",
    # Transforms
    "compute_normalization_stats",
    "Normalize",
    "NormalizeBy",
    "RandomCropTS",
    "CenterCropTS",
    "ComposeTS",
]
