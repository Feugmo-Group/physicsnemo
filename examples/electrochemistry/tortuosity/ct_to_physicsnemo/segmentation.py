"""Segmentation: Otsu baseline; optional U-Net. Week 3."""

from __future__ import annotations
import numpy as np


def otsu_segment(volume: np.ndarray, pre_filter: str = "gaussian") -> np.ndarray:
    """Return binary mask: 1 = pore, 0 = solid."""
    raise NotImplementedError


def connected_pore_mask(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected pore component, drop isolated pores."""
    raise NotImplementedError


def verify_porosity(mask: np.ndarray, expected_eps: float, tol: float = 0.05) -> bool:
    """Compare mask porosity to a gravimetric expectation; log a warning if off."""
    raise NotImplementedError
