"""
Biophysical Modeling for NeuroFMX

Differentiable spiking neural networks, neuron models, and biophysical constraints.
"""

from .spiking_nets import (
    LeakyIntegrateFireNeuron,
    IzhikevichNeuron,
    HodgkinHuxleyNeuron,
    SpikingNeuralNetwork,
    SurrogateGradient,
)

from .dales_law import (
    DalesLawConstraint,
    DalesLinear,
    EINetworkClassifier,
)

from .neuron_models import (
    AdExNeuron,
    QuadraticIFNeuron,
    BiophysicalNeuronBase,
)

from .synaptic_models import (
    STDP,
    ShortTermPlasticity,
    SynapticDynamics,
)

__all__ = [
    'LeakyIntegrateFireNeuron',
    'IzhikevichNeuron',
    'HodgkinHuxleyNeuron',
    'SpikingNeuralNetwork',
    'SurrogateGradient',
    'DalesLawConstraint',
    'DalesLinear',
    'EINetworkClassifier',
    'AdExNeuron',
    'QuadraticIFNeuron',
    'BiophysicalNeuronBase',
    'STDP',
    'ShortTermPlasticity',
    'SynapticDynamics',
]
