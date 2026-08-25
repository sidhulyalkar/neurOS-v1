"""Optional interoperability adapters for established neuroscience ecosystems.

Adapters in this namespace translate external objects into stable neurOS
contracts. They intentionally live outside ``neuros-core`` so third-party
scientific stacks remain optional dependencies.
"""

from .mne import frames_from_raw, raw_from_signal_frames, stream_descriptor_from_raw

__all__ = [
    "frames_from_raw",
    "raw_from_signal_frames",
    "stream_descriptor_from_raw",
]
