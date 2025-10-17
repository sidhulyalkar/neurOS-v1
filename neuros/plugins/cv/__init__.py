"""
Computer vision plugins
======================

This subpackage houses modules for computer‑vision tasks within neurOS.  It
includes backbones for feature extraction and heads for downstream tasks such
as segmentation.  The DINOv3 backbone is provided as a placeholder
implementation which produces deterministic random features to allow for
experiment prototyping without requiring heavy dependencies like PyTorch.

To use the DINOv3 backbone and the linear segmentation head:

.. code-block:: python

    from neuros.plugins.cv.dinov3_backbone import DINOv3Backbone
    from neuros.plugins.cv.linear_seg_head import LinearSegHead

    backbone = DINOv3Backbone(model_id="cnx-tiny")
    patch_feats = backbone.embed([image_array])
    seg_head = LinearSegHead(in_dim=backbone.feature_dim, n_classes=2)
    logits = seg_head.forward(patch_feats, (backbone.grid_size, backbone.grid_size), image_array.shape[:2])

The segmentation head expects patch features of shape ``[B, N, C]`` where
``B`` is the batch size, ``N`` is the number of patches per image and ``C``
is the feature dimension.  It outputs logits at the original image resolution.
"""

from neuros.plugins.cv.dinov3_backbone import DINOv3Backbone  # noqa: F401
from neuros.plugins.cv.linear_seg_head import LinearSegHead    # noqa: F401

__all__ = ["DINOv3Backbone", "LinearSegHead"]
