"""
Causal Interventions for Mechanistic Interpretability

Comprehensive suite for experimental interventions on neural systems:

Computational Interventions:
- Activation patching and ablation
- Path analysis and information flow
- Component-wise interventions

Biophysical Interventions:
- Optogenetics (ChR2, NpHR, ArchT, ChETA, etc.)
- Pharmacology (agonists, antagonists, channel blockers)
- Neural stimulation (TMS, DBS, tDCS, electrical)

This enables testing causal hypotheses about how foundation models
process information through experimental manipulation.
"""

# Computational Interventions
from .patching import (
    ActivationPatcher,
    ResidualStreamPatcher,
    AttentionPatcher,
    MLPPatcher,
)

from .ablation import (
    NeuronAblation,
    LayerAblation,
    ComponentAblation,
    AblationStudy,
)

from .paths import (
    PathAnalyzer,
    InformationFlow,
    CausalGraph,
)

# Optogenetic Interventions
from .optogenetics import (
    Opsin,
    ChR2,
    ChR2_H134R,
    ChETA,
    ReaChR,
    NpHR,
    ArchT,
    eNpHR3,
    OptoStimulator,
    OptogeneticExperiment,
    OptogeneticParameters,
    select_opsin,
)

# Pharmacological Interventions
from .pharmacology import (
    Drug,
    DrugMechanism,
    DrugParameters,
    Drugs,
    PharmacologyExperiment,
)

# Neural Stimulation
from .stimulation import (
    TMS,
    DBS,
    ElectricalStimulation,
    TDCS,
    StimulationType,
    StimulationParameters,
    StimulationExperiment,
)

__all__ = [
    # Computational Patching
    'ActivationPatcher',
    'ResidualStreamPatcher',
    'AttentionPatcher',
    'MLPPatcher',

    # Ablation
    'NeuronAblation',
    'LayerAblation',
    'ComponentAblation',
    'AblationStudy',

    # Path Analysis
    'PathAnalyzer',
    'InformationFlow',
    'CausalGraph',

    # Optogenetics
    'Opsin',
    'ChR2',
    'ChR2_H134R',
    'ChETA',
    'ReaChR',
    'NpHR',
    'ArchT',
    'eNpHR3',
    'OptoStimulator',
    'OptogeneticExperiment',
    'OptogeneticParameters',
    'select_opsin',

    # Pharmacology
    'Drug',
    'DrugMechanism',
    'DrugParameters',
    'Drugs',
    'PharmacologyExperiment',

    # Neural Stimulation
    'TMS',
    'DBS',
    'ElectricalStimulation',
    'TDCS',
    'StimulationType',
    'StimulationParameters',
    'StimulationExperiment',
]
