from neuros.config import load_config, resolve_config


def test_mock_bci_config_resolves_to_source_and_decoder_graph():
    config = load_config("configs/examples/mock_bci.yaml")
    resolved = resolve_config(config)
    assert "source:eeg" in resolved.graph.nodes
    assert "decoder:primary" in resolved.graph.nodes
    assert len(resolved.graph.edges) == 1
    assert resolved.graph.edges[0].source == "source:eeg"
    assert resolved.graph.edges[0].target == "decoder:primary"
    assert resolved.streams[0].source.channels == 8
