"""
Data loading and preprocessing utilities for land-take prediction.

This module provides datasets, transforms, and split utilities for
training segmentation models on HABLOSS satellite imagery.
"""

from src.data.habloss_dataset import HablossSampleDataset
from src.data.sentinel_dataset import SentinelDataset
from src.data.tessera_dataset import TesseraDataset
from src.data.wrap_datasets import FusedDataset, FusedSentinelTesseraDataset
from src.data.splits import get_splits, get_ref_ids_from_directory
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
    "HablossSampleDataset",
    "SentinelDataset",
    "TesseraDataset",
    "FusedDataset",
    "FusedSentinelTesseraDataset",
    # Splits
    "get_splits",
    "get_ref_ids_from_directory",
    # Transforms
    "compute_normalization_stats",
    "Normalize",
    "NormalizeBy",
    "RandomCropTS",
    "CenterCropTS",
    "ComposeTS",
]
