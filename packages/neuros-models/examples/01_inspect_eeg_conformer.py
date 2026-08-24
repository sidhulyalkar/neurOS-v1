"""Inspect the model-side contract without running a mech-int experiment."""

from neuros.models import EEGConformerModel, validate_manifest_paths

model = EEGConformerModel(
    n_channels=8,
    n_classes=2,
    embedding_dim=32,
    n_heads=4,
    n_layers=2,
    pool_length=16,
    pool_stride=8,
    n_epochs=1,
)
manifest = model.analysis_manifest()
print(manifest.architecture)
print("fingerprint:", manifest.fingerprint())
print("mechint-ready:", manifest.mechint_ready)
print("missing paths:", validate_manifest_paths(model.analysis_model(), manifest))
for surface in manifest.surfaces:
    print(f"{surface.path:36} {surface.role:30} {surface.recommended_methods}")
