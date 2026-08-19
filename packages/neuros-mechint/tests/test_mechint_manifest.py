from dataclasses import dataclass

import numpy as np
import torch
from neuros.quality import BenchmarkManifest

from neuros_mechint import EvidenceTier, ExperimentManifest, stable_hash


@dataclass(frozen=True)
class Nested:
    values: np.ndarray


def test_full_content_hash_distinguishes_same_shape_values():
    left = np.zeros((2, 3), dtype=np.float32)
    right = left.copy()
    right[0, 0] = 1.0
    assert stable_hash(left) != stable_hash(right)
    assert stable_hash(torch.tensor(left)) != stable_hash(torch.tensor(right))


def test_tensor_hash_handles_bfloat16_and_scalars():
    assert stable_hash(torch.tensor(1.0)) != stable_hash(torch.tensor(2.0))
    assert stable_hash(torch.tensor([1.0], dtype=torch.bfloat16))
    assert stable_hash(Nested(np.array([1, 2, 3], dtype=np.int64)))


def test_manifest_composes_neuros_benchmark_provenance():
    manifest = ExperimentManifest(
        experiment_name="manifest-test",
        method="unit",
        model_id="model",
        dataset_hash="abc",
        parameters={"x": 1},
        evidence_tier=EvidenceTier.CONTRACT,
    )
    assert isinstance(manifest.benchmark, BenchmarkManifest)
    payload = manifest.to_dict()
    assert payload["evidence_tier"] == {"level": 2, "label": "contract"}
    assert payload["benchmark"]["data_hash"] == "abc"
    assert len(manifest.content_hash) == 64
