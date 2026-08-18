from neuros.plugins import PluginKind, PluginRegistry


def test_installed_bci_profile_discovers_mock_source_and_simple_decoder():
    registry = PluginRegistry()
    registry.discover([PluginKind.SOURCE, PluginKind.DECODER])
    sources = {item.name for item in registry.list(PluginKind.SOURCE)}
    decoders = {item.name for item in registry.list(PluginKind.DECODER)}
    assert "mock" in sources
    assert "simple" in decoders

    source = registry.create("source", "mock", sampling_rate=250.0, channels=4)
    decoder = registry.create("decoder", "simple")
    assert source.channels == 4
    assert decoder.is_trained is False
