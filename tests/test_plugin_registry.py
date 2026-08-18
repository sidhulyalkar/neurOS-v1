from neuros.plugins import PluginKind, PluginRegistry


def test_registry_register_and_create():
    registry = PluginRegistry()
    registry.register(
        name="constant",
        kind=PluginKind.DECODER,
        factory=lambda value=3: {"value": value},
    )
    assert registry.create("decoder", "constant", value=7) == {"value": 7}


def test_registry_rejects_duplicate_names_within_kind():
    registry = PluginRegistry()
    registry.register(name="x", kind="source", factory=lambda: 1)
    try:
        registry.register(name="x", kind="source", factory=lambda: 2)
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate plugin should fail")
