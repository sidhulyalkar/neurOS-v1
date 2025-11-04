"""
Biophysical Modeling for NeuroFMX

Comprehensive biophysical modeling suite for mechanistic interpretability:
- Spiking neuron models (LIF, Izhikevich, Hodgkin-Huxley, AdEx, etc.)
- Ion channel dynamics (Na, K, Ca, HCN, AMPA, NMDA, GABA)
- Compartmental modeling (dendrites, soma, cable theory)
- Synaptic plasticity (STDP, STP, BCM, homeostatic)
- Metabolic constraints (ATP dynamics, energy efficiency)
- Dale's law (E/I separation)

This enables understanding foundation models through biophysically realistic
neural dynamics that mirror actual brain computation.
"""

# Spiking Neural Networks
from .spiking_nets import (
    LeakyIntegrateFireNeuron,
    IzhikevichNeuron,
    HodgkinHuxleyNeuron,
    SpikingNeuralNetwork,
    SurrogateGradient,
)

# Dale's Law and E/I Networks
from .dales_law import (
    DalesLawConstraint,
    DalesLinear,
    EINetworkClassifier,
)

# Extended Neuron Models
from .neuron_models import (
    AdExNeuron,
    QuadraticIFNeuron,
    ResonateAndFireNeuron,
    PinskyRinzelNeuron,
    BiophysicalNeuronBase,
    NeuronParameters,
)

# Synaptic Plasticity
from .synaptic_models import (
    STDP,
    TripletSTDP,
    ShortTermPlasticity,
    SynapticDynamics,
    HomeostaticPlasticity,
    BCMRule,
    Metaplasticity,
    STDPParameters,
)

# Ion Channels
from .ion_channels import (
    VoltageGatedChannel,
    SodiumChannel,
    PotassiumChannel,
    ATypeKChannel,
    CalciumChannel,
    HCNChannel,
    LigandGatedChannel,
    AMPAReceptor,
    NMDAReceptor,
    GABAAReceptor,
    GABABReceptor,
    ChannelPopulation,
    ChannelKinetics,
)

# Compartmental Modeling
from .compartmental import (
    Compartment,
    MultiCompartmentNeuron,
    CompartmentGeometry,
    CableProperties,
    PrefabNeurons,
    DendriticComputationAnalyzer,
)

# Metabolic Constraints
from .metabolic import (
    ATPDynamics,
    MetabolicConstraint,
    EnergyEfficiencyAnalyzer,
    GlucoseTransport,
    EnergyBudget,
)

__all__ = [
    # Spiking Networks
    'LeakyIntegrateFireNeuron',
    'IzhikevichNeuron',
    'HodgkinHuxleyNeuron',
    'SpikingNeuralNetwork',
    'SurrogateGradient',

    # Dale's Law
    'DalesLawConstraint',
    'DalesLinear',
    'EINetworkClassifier',

    # Extended Neuron Models
    'AdExNeuron',
    'QuadraticIFNeuron',
    'ResonateAndFireNeuron',
    'PinskyRinzelNeuron',
    'BiophysicalNeuronBase',
    'NeuronParameters',

    # Synaptic Plasticity
    'STDP',
    'TripletSTDP',
    'ShortTermPlasticity',
    'SynapticDynamics',
    'HomeostaticPlasticity',
    'BCMRule',
    'Metaplasticity',
    'STDPParameters',

    # Ion Channels
    'VoltageGatedChannel',
    'SodiumChannel',
    'PotassiumChannel',
    'ATypeKChannel',
    'CalciumChannel',
    'HCNChannel',
    'LigandGatedChannel',
    'AMPAReceptor',
    'NMDAReceptor',
    'GABAAReceptor',
    'GABABReceptor',
    'ChannelPopulation',
    'ChannelKinetics',

    # Compartmental Modeling
    'Compartment',
    'MultiCompartmentNeuron',
    'CompartmentGeometry',
    'CableProperties',
    'PrefabNeurons',
    'DendriticComputationAnalyzer',

    # Metabolic Constraints
    'ATPDynamics',
    'MetabolicConstraint',
    'EnergyEfficiencyAnalyzer',
    'GlucoseTransport',
    'EnergyBudget',
]
