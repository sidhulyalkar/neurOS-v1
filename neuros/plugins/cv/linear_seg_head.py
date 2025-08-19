"""
Linear segmentation head
=======================

This module defines a lightweight linear segmentation head suitable for use
with patch features extracted from the DINOv3 backbone.  It implements a
simple 1×1 convolution followed by bilinear interpolation to upsample the
features back to the original image resolution.  Although the name implies
segmentation, the head can be used for any dense prediction task by
adjusting the number of output classes.

The implementation is deliberately dependency‑free: it uses only NumPy to
perform its computations.  It should not be considered an efficient
replacement for deep learning libraries but provides enough functionality
for small experiments and for verifying the plumbing between feature
extraction and segmentation.  In a production setting, you would replace
this with a PyTorch or TensorFlow module.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple


class LinearSegHead:
    """A simple segmentation head for converting patch features to pixel logits.

    Parameters
    ----------
    in_dim: int
        Dimensionality of the input patch features ``C``.
    n_classes: int
        Number of output classes.  The head will produce ``n_classes``
        channels, one for each class.

    Notes
    -----
    This implementation stores a weight matrix of shape ``(n_classes, in_dim)``
    and a bias vector of shape ``(n_classes,)``.  During the forward pass,
    patch features of shape ``(B, N, C)`` are projected to logits of shape
    ``(B, N, n_classes)`` using matrix multiplication.  The result is then
    reshaped to ``(B, grid_h, grid_w, n_classes)`` and bilinearly
    interpolated back to the full image resolution.
    """

    def __init__(self, in_dim: int, n_classes: int) -> None:
        self.in_dim = in_dim
        self.n_classes = n_classes
        # Initialize weights with small random numbers
        rng = np.random.default_rng(0)
        self.weight = rng.normal(loc=0.0, scale=0.1, size=(n_classes, in_dim)).astype(np.float32)
        self.bias = np.zeros((n_classes,), dtype=np.float32)

    def forward(
        self,
        patch_feats: np.ndarray,
        patch_grid_hw: Tuple[int, int],
        out_hw: Tuple[int, int],
    ) -> np.ndarray:
        """Project patch features to pixel logits and upsample.

        Parameters
        ----------
        patch_feats: np.ndarray
            Feature tensor of shape ``(B, N, C)`` where ``B`` is the batch size,
            ``N`` is the number of patches per image and ``C`` equals
            ``in_dim``.
        patch_grid_hw: tuple of int
            The height and width of the patch grid, i.e. number of patches along
            each spatial dimension.  Typically ``(grid_size, grid_size)``.
        out_hw: tuple of int
            The height and width of the original input images.  The logits will
            be resized to this spatial resolution using bilinear interpolation.

        Returns
        -------
        numpy.ndarray
            A tensor of shape ``(B, n_classes, H, W)`` containing unnormalised
            logits at each pixel location.
        """
        B, N, C = patch_feats.shape
        assert C == self.in_dim, f"Expected feature dimension {self.in_dim}, got {C}"
        # Linear projection: (B, N, C) x (n_classes, C)^T -> (B, N, n_classes)
        logits = np.dot(patch_feats, self.weight.T) + self.bias
        # Reshape to spatial grid: (B, H_p, W_p, n_classes)
        H_p, W_p = patch_grid_hw
        logits = logits.reshape(B, H_p, W_p, self.n_classes)
        # Permute to (B, n_classes, H_p, W_p)
        logits = np.transpose(logits, (0, 3, 1, 2))
        # Upsample to (B, n_classes, H, W)
        H_out, W_out = out_hw
        upsampled = np.zeros((B, self.n_classes, H_out, W_out), dtype=np.float32)
        # Simple bilinear interpolation using NumPy
        # Compute coordinates in patch grid for each pixel
        xs = np.linspace(0, W_p - 1, num=W_out)
        ys = np.linspace(0, H_p - 1, num=H_out)
        x0 = np.floor(xs).astype(int)
        y0 = np.floor(ys).astype(int)
        x1 = np.clip(x0 + 1, 0, W_p - 1)
        y1 = np.clip(y0 + 1, 0, H_p - 1)
        wx = xs - x0
        wy = ys - y0
        for b in range(B):
            for c in range(self.n_classes):
                for j in range(H_out):
                    for i in range(W_out):
                        # Bilinear interpolation
                        y0j, y1j = y0[j], y1[j]
                        x0i, x1i = x0[i], x1[i]
                        w_x, w_y = wx[i], wy[j]
                        v00 = logits[b, c, y0j, x0i]
                        v01 = logits[b, c, y0j, x1i]
                        v10 = logits[b, c, y1j, x0i]
                        v11 = logits[b, c, y1j, x1i]
                        upsampled[b, c, j, i] = (
                            v00 * (1 - w_x) * (1 - w_y)
                            + v01 * w_x * (1 - w_y)
                            + v10 * (1 - w_x) * w_y
                            + v11 * w_x * w_y
                        )
        return upsampled
