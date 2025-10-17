"""
Placeholder DINOv3 backbone
===========================

This module provides a drop‑in replacement for the DINOv3 convolutional and
Vision Transformer backbones described in the DINOv3 paper.  The purpose of
this implementation is to allow researchers to prototype pipelines within
neurOS without pulling in heavyweight dependencies such as PyTorch or
transformers.  It generates deterministic pseudo‑random features based on
image content so that experiments can be run end‑to‑end in restricted
environments.

**Note:** This is **not** a faithful implementation of the DINOv3 models.  It
does not produce the same features as the real models.  To integrate the
actual models, install the ``transformers`` and ``timm`` libraries and
replace the ``embed`` method with calls to ``AutoImageProcessor`` and
``AutoModel`` as shown in the accompanying notebook.
"""

from __future__ import annotations

import numpy as np
from typing import Iterable, List, Tuple, Union


class DINOv3Backbone:
    """Simulated DINOv3 backbone for neurOS.

    Parameters
    ----------
    model_id: str
        A string identifying the backbone variant.  Common values include
        ``"cnx-tiny"``, ``"cnx-small"``, ``"cnx-base"``, ``"cnx-large"``,
        ``"vit-small"``, ``"vit-base"``, ``"vit-large"``.  The feature
        dimension is derived from this identifier.
    device: str, optional
        Included for API compatibility with real backbones.  Ignored in this
        implementation.

    Attributes
    ----------
    model_id: str
        The identifier passed at construction.
    feature_dim: int
        The dimensionality of the output patch features.  Determined from
        ``model_id``.
    patch_size: int
        The size of each square patch in pixels.  Fixed at 16 to mimic
        ViT/ConvNeXt patch embeddings.
    grid_size: int
        The number of patches along the shorter side of the input images.  This
        value is populated after the first call to :meth:`embed`.
    """

    def __init__(self, model_id: str = "cnx-tiny", device: str = "cpu") -> None:
        self.model_id = model_id.lower()
        self.feature_dim = self._infer_feature_dim(self.model_id)
        self.device = device
        self.patch_size = 16
        # grid_size will be set when embed is first called
        self.grid_size: int | None = None

    def _infer_feature_dim(self, model_id: str) -> int:
        # Derive a plausible feature dimension based on the name of the model.
        # These numbers follow typical hidden sizes for ConvNeXt/Vision Transformers.
        if "cnx" in model_id or "convnext" in model_id:
            if "tiny" in model_id:
                return 384
            elif "small" in model_id or model_id.endswith("-s"):
                return 512
            elif "base" in model_id or model_id.endswith("-b"):
                return 768
            elif "large" in model_id or model_id.endswith("-l"):
                return 1024
            elif "huge" in model_id or model_id.endswith("-h"):
                return 1280
        if "vit" in model_id:
            if "tiny" in model_id or model_id.endswith("-t"):
                return 192
            elif "small" in model_id or model_id.endswith("-s"):
                return 384
            elif "base" in model_id or model_id.endswith("-b"):
                return 768
            elif "large" in model_id or model_id.endswith("-l"):
                return 1024
            elif "huge" in model_id or model_id.endswith("-h"):
                return 1280
        # Default fallback
        return 512

    def embed(self, images: Iterable[Union[np.ndarray, 'PIL.Image.Image']]) -> np.ndarray:
        """Compute pseudo‑random patch features for a batch of images.

        Parameters
        ----------
        images: iterable of array-like
            A collection of images.  Each element may be a NumPy array of shape
            ``(H, W, 3)`` or a PIL.Image.  The colour channels must be in RGB
            order with pixel values in ``[0, 255]``.  Images with spatial
            dimensions not divisible by the patch size will be cropped at the
            bottom and right edges.

        Returns
        -------
        numpy.ndarray
            A 3D array of shape ``(B, N, C)`` where ``B`` is the batch size,
            ``N`` is the number of patches per image (equal to
            ``grid_size * grid_size``), and ``C`` is the feature dimension.

        Notes
        -----
        The features are deterministic for a given image and model_id.  They
        encode the mean pixel intensity per patch plus a small amount of
        pseudo‑random noise seeded by the patch index.  This ensures that
        different model identifiers produce distinct but reproducible feature
        vectors.
        """
        batch_features: List[np.ndarray] = []
        for img in images:
            if hasattr(img, 'convert'):
                # PIL.Image instance
                arr = np.array(img.convert("RGB"), dtype=np.float32)
            else:
                arr = np.array(img, dtype=np.float32)
            # Ensure three channels
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            elif arr.ndim == 3 and arr.shape[2] == 1:
                arr = np.concatenate([arr] * 3, axis=-1)
            H, W, _ = arr.shape
            # Crop to multiples of patch_size
            ph = H // self.patch_size
            pw = W // self.patch_size
            cropped = arr[:ph * self.patch_size, :pw * self.patch_size]
            # Compute mean intensity per patch (over spatial dimensions and channels)
            patches = cropped.reshape(ph, self.patch_size, pw, self.patch_size, 3)
            patch_means = patches.mean(axis=(1, 3, 4))  # shape (ph, pw)
            # Determine grid size on first call
            if self.grid_size is None:
                self.grid_size = ph  # assume square inputs; we only use ph
            # Flatten patches and compute features
            feats = []
            for idx, val in enumerate(patch_means.flatten()):
                # Seed derived from model_id and patch index
                seed = (hash(self.model_id) + idx * 997) % 2**32
                rng = np.random.default_rng(seed)
                noise = rng.normal(loc=0.0, scale=0.05, size=self.feature_dim)
                feat = np.full(self.feature_dim, val, dtype=np.float32) + noise
                feats.append(feat.astype(np.float32))
            feats_arr = np.stack(feats, axis=0)  # (N, C)
            batch_features.append(feats_arr)
        return np.stack(batch_features, axis=0)  # (B, N, C)
