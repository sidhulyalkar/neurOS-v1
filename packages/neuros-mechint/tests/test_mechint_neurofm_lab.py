import numpy as np
import pytest
import torch
from torch import nn

pytest.importorskip("orion")
from orion.contracts import RepresentationBatch

from neuros_mechint.benchmarks import MechanismContext
from neuros_mechint.integrations.neurofm import (
    NeuroFMCheckpointContext,
    NeuroFMProbeSpec,
    NeuroFMRepresentationProbe,
    model_call,
    run_neurofm_mechanism_lab,
)


class TinyNeuroFM(nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.backbone = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.backbone.weight.copy_(torch.eye(2) * scale)

    def forward(self, tokens: torch.Tensor, attention_mask=None):
        if attention_mask is not None:
            tokens = tokens * attention_mask.unsqueeze(-1)
        return self.backbone(tokens)


class CompressedNeuroFM(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Identity()
        self.compressed = _Compressor()

    def forward(self, tokens: torch.Tensor):
        return self.compressed(self.backbone(tokens))


class _Compressor(nn.Module):
    def forward(self, values: torch.Tensor):
        return values[:, ::2]


class PooledNeuroFM(nn.Module):
    def __init__(self):
        super().__init__()
        self.pooled = _PoolTime()

    def forward(self, tokens: torch.Tensor):
        return self.pooled(tokens)


class _PoolTime(nn.Module):
    def forward(self, values: torch.Tensor):
        return values.mean(dim=1)


def _checkpoint(architecture: str, step: int, scale: float):
    tokens = torch.tensor(
        [[[1.0, 0.1], [2.0, 0.2], [4.0, 0.4], [0.5, 0.05]]],
        dtype=torch.float32,
    )
    mask = torch.ones((1, 4), dtype=torch.float32)

    def scorer(batch: RepresentationBatch) -> float:
        return float(np.asarray(batch.values)[:, 0].sum())

    return NeuroFMCheckpointContext(
        context=MechanismContext(
            context_id=f"{architecture}-{step}",
            architecture=architecture,
            dataset_id="synthetic-neural",
            session_id="s1",
            subject_id="mouse-1",
            checkpoint=f"step:{step}",
        ),
        training_step=step,
        model=TinyNeuroFM(scale),
        model_inputs=model_call(tokens, attention_mask=mask),
        input_timestamps_ns=np.asarray([990, 1000, 1010, 1020], dtype=np.int64),
        scorer=scorer,
        alignment_origin_ns=1000,
        alignment_label="stimulus_onset",
    )


def test_neurofm_lab_captures_states_and_builds_checkpoint_trajectory():
    result = run_neurofm_mechanism_lab(
        [
            _checkpoint("ssm", 0, 0.2),
            _checkpoint("ssm", 100, 0.6),
            _checkpoint("ssm", 200, 1.0),
            _checkpoint("transformer", 200, 0.9),
        ],
        window_ns=10,
        stride_ns=10,
        top_k=2,
    )
    assert "ssm|synthetic-neural|s1|mouse-1" in result.emergence_reports
    assert result.emergence_reports["ssm|synthetic-neural|s1|mouse-1"].final_step == 200
    architecture = result.shared_study.analysis.comparison.isolated_axis_stability[
        "architecture"
    ]
    assert architecture.pair_count == 1
    assert result.checkpoint_steps["ssm-200"] == 200


def test_neurofm_probe_requires_explicit_timestamps_for_compressed_states():
    model = CompressedNeuroFM()
    probe = NeuroFMRepresentationProbe(
        model,
        NeuroFMProbeSpec(component_path="compressed"),
    )
    tokens = torch.ones((1, 4, 2), dtype=torch.float32)
    with pytest.raises(ValueError, match="representation_timestamps_ns"):
        probe.capture(tokens, input_timestamps_ns=[0, 10, 20, 30])

    captured = probe.capture(
        tokens,
        input_timestamps_ns=[0, 10, 20, 30],
        representation_timestamps_ns=[0, 20],
    )
    assert captured.values.shape == (2, 2)
    assert captured.timestamps_ns.tolist() == [0, 20]


def test_neurofm_probe_does_not_confuse_batched_pooled_state_with_time():
    model = PooledNeuroFM()
    probe = NeuroFMRepresentationProbe(
        model,
        NeuroFMProbeSpec(component_path="pooled"),
    )
    tokens = torch.ones((1, 4, 2), dtype=torch.float32)
    with pytest.raises(ValueError, match="representation_timestamps_ns"):
        probe.capture(tokens, input_timestamps_ns=[0, 10, 20, 30])

    captured = probe.capture(
        tokens,
        input_timestamps_ns=[0, 10, 20, 30],
        representation_timestamps_ns=[15],
    )
    assert captured.values.shape == (1, 2)
    assert captured.timestamps_ns.tolist() == [15]
