import asyncio
import importlib.metadata

import numpy as np
import pytest

from neuros.contracts import SignalFrame, StreamDescriptor, WindowSpec
from neuros.models import BraindecodeDecoder
from neuros.processing import SlidingWindowTransform
from neuros.runtime import NodeKind, RuntimeEdge, RuntimeExecutor, RuntimeGraph, RuntimeNode


def test_braindecode_adapter_rejects_unqualified_or_ambiguous_contracts():
    with pytest.raises(ValueError, match="Qualified adapter models"):
        BraindecodeDecoder("SignalJEPA", 2, 128, 2)

    with pytest.raises(ValueError, match="cannot override"):
        BraindecodeDecoder(
            "EEGNet",
            2,
            128,
            2,
            model_options={"n_chans": 99},
        )

    with pytest.raises(ValueError, match="JSON-serializable"):
        BraindecodeDecoder(
            "EEGNet",
            2,
            128,
            2,
            model_options={"activation": object()},
        )

    decoder = BraindecodeDecoder("EEGNet", 2, 128, 2, n_epochs=1)
    with pytest.raises(ValueError, match="batch, channels, time"):
        decoder.train(np.ones((2, 128), dtype=np.float32), np.array([0, 1]))
    with pytest.raises(ValueError, match="geometry"):
        decoder.train(np.ones((2, 3, 128), dtype=np.float32), np.array([0, 1]))


def test_adapter_configuration_fingerprint_is_stable_and_sensitive():
    first = BraindecodeDecoder(
        "EEGNet",
        2,
        128,
        2,
        sample_rate_hz=128.0,
        n_epochs=1,
        random_state=7,
    )
    same = BraindecodeDecoder(
        "eeg_net",
        2,
        128,
        2,
        sample_rate_hz=128.0,
        n_epochs=1,
        random_state=7,
    )
    changed = BraindecodeDecoder(
        "EEGNet",
        2,
        128,
        2,
        sample_rate_hz=128.0,
        n_epochs=1,
        random_state=8,
    )

    assert first.configuration() == same.configuration()
    assert first.configuration_fingerprint == same.configuration_fingerprint
    assert first.configuration_fingerprint != changed.configuration_fingerprint
    assert len(first.configuration_fingerprint) == 16


@pytest.mark.parametrize(
    "model_name",
    ["EEGNet", "EEGConformer", "ShallowFBCSPNet", "Deep4Net"],
)
def test_qualified_braindecode_models_share_the_neuros_window_geometry(model_name):
    import torch

    decoder = BraindecodeDecoder(
        model_name,
        n_channels=4,
        n_times=512,
        n_classes=3,
        sample_rate_hz=128.0,
        n_epochs=1,
        batch_size=2,
        device="cpu",
    )
    module = decoder._build_module()
    module.eval()
    with torch.no_grad():
        output = module(torch.zeros(2, 4, 512, dtype=torch.float32))
    if isinstance(output, tuple):
        output = output[0]
    assert tuple(output.shape) == (2, 3)


class _OneWindowSource:
    def __init__(self, sample_major: np.ndarray) -> None:
        self.sample_major = np.asarray(sample_major, dtype=np.float32)
        self._descriptor = StreamDescriptor(
            stream_id="braindecode-eeg",
            modality="eeg",
            sample_rate_hz=128.0,
            channel_names=("C3", "C4"),
        )

    @property
    def descriptor(self) -> StreamDescriptor:
        return self._descriptor

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def frames(self):
        yield SignalFrame(
            stream_id=self._descriptor.stream_id,
            sequence_id=0,
            data=self.sample_major,
            sample_rate_hz=self._descriptor.sample_rate_hz,
            host_receive_time_ns=1_000_000_000,
            metadata={
                "axis_order": ("sample", "channel"),
                "channel_names": self._descriptor.channel_names,
            },
        )
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_braindecode_eegnet_runs_through_neuros_window_runtime_without_hidden_preprocessing():
    version = importlib.metadata.version("braindecode")
    assert version.startswith("1.7."), version

    rng = np.random.default_rng(11)
    X = rng.normal(size=(8, 2, 128)).astype(np.float32)
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)

    decoder = BraindecodeDecoder(
        "EEGNet",
        n_channels=2,
        n_times=128,
        n_classes=2,
        sample_rate_hz=128.0,
        learning_rate=1e-3,
        n_epochs=1,
        batch_size=4,
        device="cpu",
        random_state=11,
    )
    decoder.train(X, y)

    direct = decoder.infer(X[:1])
    assert direct.model_id == "Braindecode:EEGNet"
    assert direct.model_version == version
    assert direct.probabilities is not None
    probabilities = np.asarray(direct.probabilities)
    assert probabilities.shape == (2,)
    assert np.isfinite(probabilities).all()
    assert np.isclose(probabilities.sum(), 1.0, atol=1e-5)
    assert direct.confidence is not None
    assert direct.metadata["backend"] == "braindecode/torch"
    assert direct.metadata["upstream_training"] == "EEGClassifier"
    assert direct.metadata["hidden_preprocessing"] is False
    assert direct.metadata["input_contract"] == "batch,channel,time"
    assert direct.metadata["adapter_config_fingerprint"] == decoder.configuration_fingerprint
    assert direct.metadata["random_state"] == 11

    source = _OneWindowSource(X[0].T)
    graph = RuntimeGraph()
    graph.add_node(RuntimeNode("source", NodeKind.SOURCE, source))
    graph.add_node(
        RuntimeNode(
            "window",
            NodeKind.TRANSFORM,
            SlidingWindowTransform(WindowSpec(128, 128), descriptor=source.descriptor),
        )
    )
    graph.add_node(RuntimeNode("decoder", NodeKind.DECODER, decoder))
    graph.connect(RuntimeEdge("source", "window", overflow="block"))
    graph.connect(RuntimeEdge("window", "decoder", overflow="block"))

    executor = RuntimeExecutor(graph)
    await executor.start()
    outputs = [output async for output in executor.outputs()]
    await executor.wait()

    assert len(outputs) == 1
    output = outputs[0]
    assert output.model_id == "Braindecode:EEGNet"
    assert output.metadata["neuros_stream_id"] == "braindecode-eeg"
    assert output.metadata["neuros_window_id"] == 0
    assert output.metadata["window_channel_names"] == ("C3", "C4")
    assert output.metadata["source_sequence_ids"] == (0,)
    assert output.metadata["backend"] == "braindecode/torch"
    assert output.metadata["hidden_preprocessing"] is False
    assert output.metadata["adapter_config_fingerprint"] == decoder.configuration_fingerprint

    module = decoder.analysis_model()
    assert module is not None
    manifest = decoder.analysis_manifest()
    assert manifest.architecture == "EEGNet"
    assert manifest.input_axes == ("batch", "channel", "time")
    assert manifest.mechint_ready is False
