"""Run ZUNA1.1 through the verified upstream zuna package."""

from neuros.foundation_models import DEFAULT_REGISTRY

zuna = DEFAULT_REGISTRY.adapter("zuna-1.1")
zuna.reconstruct_fif(
    input_dir="recordings/input",
    output_dir="recordings/output",
    figures_dir="recordings/figures",
    montage="standard_1020",
    gpu_device="",  # CPU; set an upstream-supported GPU id when configured.
)
