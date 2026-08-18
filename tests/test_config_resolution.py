from neuros.config import load_config, resolve_config


def test_mock_bci_config_resolves_to_executable_source_transform_decoder_graph():
    config = load_config("configs/examples/mock_bci.yaml")
    resolved = resolve_config(config)
    assert set(resolved.graph.nodes) == {
        "source:eeg",
        "transform:eeg:0",
        "decoder:primary",
    }
    edges = {(edge.source, edge.target) for edge in resolved.graph.edges}
    assert edges == {
        ("source:eeg", "transform:eeg:0"),
        ("transform:eeg:0", "decoder:primary"),
    }
    assert resolved.streams[0].source.channels == 8
    assert hasattr(resolved.streams[0].transforms[0], "transform")
