# BCI Notebooks

`notebooks/` is a transitional learning/research surface for BCI material that has not yet been promoted to a maintained tutorial or supported example.

The clearly DINOv3-specific research notebooks were moved to `experiments/vision/dinov3/` so this directory no longer mixes unrelated computer-vision research with BCI learning material.

## Promotion

A notebook should move to:

- `tutorials/` when it is maintained educational material using current APIs;
- `examples/` when it is an executable supported user workflow with a CI smoke test;
- `experiments/` when it is exploratory research or depends on specialized datasets/compute.

Do not assume a notebook in this directory is part of the stable neurOS API. Check current package docs and tests when adapting historical examples.
