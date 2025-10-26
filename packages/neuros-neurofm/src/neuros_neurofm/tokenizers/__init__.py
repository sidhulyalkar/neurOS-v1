"""
Neural tokenizers for NeuroFM-X.

Tokenizers convert raw neural data into discrete or continuous tokens
suitable for processing by the Mamba/SSM backbone.

Supports multiple modalities:
- Electrophysiology: Spikes, LFP, ECoG
- Calcium imaging: 2-photon, Miniscope
- Human recordings: EEG, fMRI
- Motor: EMG

New features:
- TokenizedSequence: Unified temporal representation
- TemporalAligner: Multi-modal synchronization
- BaseTokenizer: Common interface for all tokenizers
"""

# Base classes and data structures
from neuros_neurofm.tokenizers.base_tokenizer import (
    BaseTokenizer,
    TokenizedSequence,
    concatenate_sequences,
    batch_sequences,
)

# Temporal alignment utilities
from neuros_neurofm.tokenizers.temporal_alignment import (
    TemporalAligner,
    InterpolationMethod,
    resample_to_rate,
    align_and_concatenate,
)

# Modality-specific tokenizers
from neuros_neurofm.tokenizers.spike_tokenizer import SpikeTokenizer
from neuros_neurofm.tokenizers.lfp_tokenizer import LFPTokenizer
from neuros_neurofm.tokenizers.binned_tokenizer import BinnedTokenizer
from neuros_neurofm.tokenizers.calcium_tokenizer import (
    CalciumTokenizer,
    TwoPhotonTokenizer,
    MiniscopeTokenizer,
)
from neuros_neurofm.tokenizers.eeg_tokenizer import EEGTokenizer
from neuros_neurofm.tokenizers.fmri_tokenizer import fMRITokenizer

__all__ = [
    # Base classes
    "BaseTokenizer",
    "TokenizedSequence",
    "concatenate_sequences",
    "batch_sequences",
    # Temporal alignment
    "TemporalAligner",
    "InterpolationMethod",
    "resample_to_rate",
    "align_and_concatenate",
    # Tokenizers
    "SpikeTokenizer",
    "LFPTokenizer",
    "BinnedTokenizer",
    "CalciumTokenizer",
    "TwoPhotonTokenizer",
    "MiniscopeTokenizer",
    "EEGTokenizer",
    "fMRITokenizer",
]
