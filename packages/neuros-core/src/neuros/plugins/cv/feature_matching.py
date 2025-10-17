"""
Patch correlation and registration utilities
==========================================

This module implements simple feature matching utilities for evaluating
representation quality in registration and correspondence tasks.  It is not a
replacement for sophisticated optical flow or graph matching algorithms but
provides a baseline that can be used to test whether features capture
geometric relationships between images.

The functions are designed to work with patch features extracted from the
DINOv3 backbone.  They assume that each image is divided into a grid of
patches of equal size and that patches are indexed in row-major order.  All
functions are deterministic and rely only on NumPy.

"""

from __future__ import annotations

import numpy as np
from typing import Tuple


def patch_correlation(features_a: np.ndarray, features_b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between two sets of patch features.

    Parameters
    ----------
    features_a: numpy.ndarray
        Patch features of shape ``(N_a, C)`` extracted from the first image.
    features_b: numpy.ndarray
        Patch features of shape ``(N_b, C)`` extracted from the second image.

    Returns
    -------
    numpy.ndarray
        A similarity matrix of shape ``(N_a, N_b)`` where each entry `(i, j)`
        contains the cosine similarity between patch ``i`` of image A and patch
        ``j`` of image B.
    """
    assert features_a.ndim == 2 and features_b.ndim == 2, "Input features must be 2D"
    # Normalise the feature vectors to unit length
    norms_a = np.linalg.norm(features_a, axis=1, keepdims=True) + 1e-8
    norms_b = np.linalg.norm(features_b, axis=1, keepdims=True) + 1e-8
    a_norm = features_a / norms_a
    b_norm = features_b / norms_b
    # Cosine similarity via dot product
    sim = a_norm @ b_norm.T
    return sim


def estimate_translation(similarity: np.ndarray, grid_size: int) -> Tuple[int, int]:
    """Estimate translation between two patch grids from a similarity matrix.

    Given a similarity matrix between patches of two images, this function
    identifies the best-matching patch in the second image for each patch in
    the first image and computes the median offset between their grid
    coordinates.  The result is a discrete translation measured in units of
    patches.

    Parameters
    ----------
    similarity: numpy.ndarray
        Cosine similarity matrix of shape ``(N, N)`` where ``N`` is the number of
        patches per image (must be the same for both images).
    grid_size: int
        The number of patches along one dimension (assuming square grid).

    Returns
    -------
    (dy, dx): tuple of int
        Estimated vertical and horizontal translation between the two patch grids
        measured in patch units.  A positive ``dy`` indicates that the second
        image is shifted downward relative to the first; a positive ``dx``
        indicates a rightward shift.
    """
    assert similarity.shape[0] == similarity.shape[1], "Similarity matrix must be square"
    N = similarity.shape[0]
    # Determine the best match in image B for each patch in image A
    best_indices = np.argmax(similarity, axis=1)
    # Convert linear indices to 2D grid coordinates
    coords_a = np.array([(i // grid_size, i % grid_size) for i in range(N)])
    coords_b = np.array([(j // grid_size, j % grid_size) for j in best_indices])
    # Compute pairwise offsets
    diffs = coords_b - coords_a
    # Use median to robustly estimate translation
    median_offset = np.median(diffs, axis=0)
    dy, dx = int(np.round(median_offset[0])), int(np.round(median_offset[1]))
    return dy, dx
