"""Typed neurOS exceptions used to classify runtime failure modes."""


class NeurOSError(Exception):
    """Base class for neurOS errors."""


class ConfigurationError(NeurOSError):
    """Invalid or unsupported runtime configuration."""


class SourceConnectionError(NeurOSError):
    """A data source could not connect or remain connected."""


class ClockSyncError(NeurOSError):
    """Clock alignment failed or became untrustworthy."""


class DataContractError(NeurOSError):
    """Data violated a declared neurOS contract."""


class ProcessingError(NeurOSError):
    """Signal processing failed."""


class DecoderError(NeurOSError):
    """Decoder inference or adaptation failed."""


class ArtifactError(NeurOSError):
    """A model or data artifact failed validation or loading."""


class SafetyViolation(NeurOSError):
    """A runtime action violated an explicit safety constraint."""
