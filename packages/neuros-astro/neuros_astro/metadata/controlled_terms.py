"""Controlled vocabulary terms for dataset triage and scoring."""

from typing import Final

# Astrocyte-related terms
ASTRO_TERMS: Final[list[str]] = [
    "astrocyte",
    "astrocytic",
    "astroglia",
    "glia",
    "glial",
    "GFAP",
    "Aldh1l1",
    "S100B",
    "SR101",
    "sulforhodamine",
    "sulphorhodamine",
]

# Calcium indicator terms
CALCIUM_TERMS: Final[list[str]] = [
    "GCaMP",
    "jGCaMP",
    "GCaMP6",
    "GCaMP7",
    "GCaMP8",
    "Cal-520",
    "Calbryte",
    "Oregon Green",
    "Oregon Green BAPTA",
    "Fluo-4",
    "Fluo-5",
    "calcium imaging",
    "Ca2+ imaging",
]

# Imaging modality terms
MODALITY_TERMS: Final[list[str]] = [
    "two-photon",
    "2-photon",
    "2p",
    "miniscope",
    "mini-microscope",
    "widefield",
    "wide-field",
    "confocal",
    "one-photon",
    "1-photon",
    "epifluorescence",
]

# Behavioral/task terms
BEHAVIOR_TERMS: Final[list[str]] = [
    "behavior",
    "behaviour",
    "running",
    "locomotion",
    "pupil",
    "whisking",
    "licking",
    "reward",
    "stimulus",
    "task",
]

# Electrophysiology terms
EPHYS_TERMS: Final[list[str]] = [
    "electrophysiology",
    "ephys",
    "spike",
    "spikes",
    "LFP",
    "local field potential",
    "neuropixels",
    "electrode",
    "extracellular",
]


def search_terms_in_text(text: str, terms: list[str]) -> list[str]:
    """
    Search for controlled terms in text (case-insensitive).

    Args:
        text: Text to search
        terms: List of terms to search for

    Returns:
        List of matched terms
    """
    if not text:
        return []

    text_lower = text.lower()
    matched = []

    for term in terms:
        if term.lower() in text_lower:
            matched.append(term)

    return matched
