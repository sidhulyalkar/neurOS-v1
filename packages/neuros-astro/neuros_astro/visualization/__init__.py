"""Visualization utilities for astrocyte events and networks."""

# Lazy imports to handle optional matplotlib dependency
__all__ = [
    "plot_event_raster",
    "plot_event_distributions",
    "plot_event_statistics_summary",
    "plot_network_graph",
    "plot_network_evolution",
    "plot_trace_with_events",
]


def __getattr__(name):
    """Lazy import of visualization functions."""
    if name in __all__:
        from neuros_astro.visualization import event_plots
        return getattr(event_plots, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
