"""
Compartmental Neuron Models

Multi-compartment models with dendritic computation, incorporating
cable theory and active conductances.

References:
- Rall (1962): Theory of physiological properties of dendrites
- Koch & Segev (1998): Methods in Neuronal Modeling
- Poirazi & Mel (2001): Impact of active dendrites
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from .ion_channels import ChannelPopulation


@dataclass
class CompartmentGeometry:
    """Geometric properties of a compartment."""
    length: float  # μm
    diameter: float  # μm
    area: float  # μm^2
    volume: float  # μm^3

    @classmethod
    def cylinder(cls, length: float, diameter: float):
        """Create cylindrical compartment geometry."""
        area = np.pi * diameter * length
        volume = np.pi * (diameter / 2) ** 2 * length
        return cls(length, diameter, area, volume)

    @classmethod
    def sphere(cls, diameter: float):
        """Create spherical compartment geometry (soma)."""
        area = np.pi * diameter ** 2
        volume = (4 / 3) * np.pi * (diameter / 2) ** 3
        return cls(0, diameter, area, volume)


@dataclass
class CableProperties:
    """Electrical properties for cable theory."""
    Ra: float = 150.0  # Axial resistance (Ω·cm)
    Cm: float = 1.0    # Membrane capacitance (μF/cm^2)
    Rm: float = 20000.0  # Membrane resistance (Ω·cm^2)

    def length_constant(self, diameter: float) -> float:
        """
        Compute electrotonic length constant.

        λ = √(Rm * d / (4 * Ra))

        Args:
            diameter: Compartment diameter (μm)

        Returns:
            Lambda: Length constant (μm)
        """
        d_cm = diameter * 1e-4  # Convert to cm
        lambda_cm = np.sqrt(self.Rm * d_cm / (4 * self.Ra))
        return lambda_cm * 1e4  # Convert back to μm

    def time_constant(self) -> float:
        """
        Membrane time constant.

        τ = Rm * Cm

        Returns:
            Tau: Time constant (ms)
        """
        return self.Rm * self.Cm


class Compartment(nn.Module):
    """
    Single compartment with ion channels and synaptic inputs.

    Represents a patch of membrane with uniform properties.
    """

    def __init__(
        self,
        geometry: CompartmentGeometry,
        channels: ChannelPopulation,
        cable_props: CableProperties,
        V_rest: float = -65.0,
        dt: float = 0.01
    ):
        super().__init__()
        self.geometry = geometry
        self.channels = channels
        self.cable_props = cable_props
        self.V_rest = V_rest
        self.dt = dt

        # State variables
        self.register_buffer('V', None)
        self.register_buffer('I_ext', None)

        # Connections to other compartments
        self.parents = []
        self.children = []
        self.coupling_conductances = []

    def initialize(self, batch_size: int, device: str = 'cpu'):
        """Initialize state variables."""
        self.V = torch.full((batch_size,), self.V_rest, device=device)
        self.I_ext = torch.zeros(batch_size, device=device)

    def axial_current(self, V_neighbor: torch.Tensor, coupling_g: float) -> torch.Tensor:
        """
        Compute axial current from neighboring compartment.

        Args:
            V_neighbor: Neighbor voltage (mV)
            coupling_g: Coupling conductance (nS)

        Returns:
            I_axial: Axial current (nA)
        """
        return coupling_g * (V_neighbor - self.V)

    def step(
        self,
        I_syn: torch.Tensor,
        I_stim: Optional[torch.Tensor] = None,
        NT: Optional[Dict[str, torch.Tensor]] = None
    ):
        """
        Single integration step using Euler method.

        Args:
            I_syn: Synaptic current (nA)
            I_stim: External stimulus current (nA)
            NT: Neurotransmitter concentrations
        """
        if self.V is None:
            raise RuntimeError("Compartment not initialized. Call initialize() first.")

        # Membrane current from channels (μA/cm^2)
        I_mem = self.channels.total_current(self.V, NT)

        # Convert to nA
        area_cm2 = self.geometry.area * 1e-8  # μm^2 to cm^2
        I_channels = I_mem * area_cm2 * 1000  # μA to nA

        # Total current
        I_total = I_syn + I_channels
        if I_stim is not None:
            I_total += I_stim

        # Axial currents from neighbors
        for parent, g_couple in zip(self.parents, self.coupling_conductances):
            I_total += self.axial_current(parent.V, g_couple)

        # dV/dt = -I_total / C
        C = self.cable_props.Cm * area_cm2  # nF
        dV = -I_total / C

        self.V = self.V + self.dt * dV

    def add_connection(self, parent: 'Compartment', coupling_g: float):
        """
        Add connection to parent compartment.

        Args:
            parent: Parent compartment
            coupling_g: Coupling conductance (nS)
        """
        self.parents.append(parent)
        self.coupling_conductances.append(coupling_g)
        parent.children.append(self)

    def reset(self):
        """Reset to resting state."""
        if self.V is not None:
            self.V.fill_(self.V_rest)
        self.channels.reset_all()


class MultiCompartmentNeuron(nn.Module):
    """
    Multi-compartment neuron model.

    Simulates realistic dendritic computation with cable theory
    and distributed ion channels.
    """

    def __init__(
        self,
        compartments: List[Compartment],
        dt: float = 0.01
    ):
        super().__init__()
        self.compartments = nn.ModuleList(compartments)
        self.dt = dt
        self.n_compartments = len(compartments)

    def forward(
        self,
        I_syn: List[torch.Tensor],
        I_stim: Optional[List[torch.Tensor]] = None,
        NT: Optional[Dict[str, torch.Tensor]] = None,
        n_steps: int = 100
    ) -> torch.Tensor:
        """
        Simulate multi-compartment neuron.

        Args:
            I_syn: List of synaptic currents for each compartment (nA)
            I_stim: List of stimulus currents for each compartment (nA)
            NT: Neurotransmitter concentrations
            n_steps: Number of time steps

        Returns:
            V_soma: Somatic voltage trace (batch, n_steps)
        """
        batch_size = I_syn[0].shape[0]
        V_soma = torch.zeros(batch_size, n_steps, device=I_syn[0].device)

        # Initialize all compartments
        for comp in self.compartments:
            if comp.V is None:
                comp.initialize(batch_size, I_syn[0].device)

        # Simulate
        for t in range(n_steps):
            # Update each compartment
            for i, comp in enumerate(self.compartments):
                stim = I_stim[i] if I_stim is not None else None
                comp.step(I_syn[i], stim, NT)

            # Record soma voltage (assume first compartment is soma)
            V_soma[:, t] = self.compartments[0].V

        return V_soma

    def reset(self):
        """Reset all compartments."""
        for comp in self.compartments:
            comp.reset()


class PrefabNeurons:
    """
    Factory for creating pre-configured multi-compartment neurons.
    """

    @staticmethod
    def compute_coupling_conductance(
        comp1: Compartment,
        comp2: Compartment,
        cable_props: CableProperties
    ) -> float:
        """
        Compute axial coupling conductance between two compartments.

        g_couple = (π / (4 * Ra)) * (d1^2 * d2^2) / (d1^2 * L2 + d2^2 * L1)

        Args:
            comp1: First compartment
            comp2: Second compartment
            cable_props: Cable properties

        Returns:
            g_couple: Coupling conductance (nS)
        """
        g1 = comp1.geometry
        g2 = comp2.geometry

        d1 = g1.diameter * 1e-4  # μm to cm
        d2 = g2.diameter * 1e-4
        L1 = g1.length * 1e-4
        L2 = g2.length * 1e-4

        if L1 == 0:
            L1 = d1  # Spherical soma

        if L2 == 0:
            L2 = d2

        Ra = cable_props.Ra  # Ω·cm

        numerator = np.pi * (d1**2) * (d2**2)
        denominator = 4 * Ra * ((d1**2) * L2 + (d2**2) * L1)

        g_couple_S = numerator / denominator  # Siemens
        return g_couple_S * 1e9  # Convert to nS

    @staticmethod
    def pyramidal_cell(
        include_channels: bool = True,
        dt: float = 0.01
    ) -> MultiCompartmentNeuron:
        """
        Create 3-compartment pyramidal neuron.

        Compartments:
        1. Soma (sphere)
        2. Basal dendrite (cylinder)
        3. Apical dendrite (cylinder)

        Args:
            include_channels: Whether to include active channels
            dt: Time step (ms)

        Returns:
            Pyramidal neuron model
        """
        from .ion_channels import (
            SodiumChannel, PotassiumChannel, ATypeKChannel,
            CalciumChannel, HCNChannel
        )

        cable_props = CableProperties()

        # Create geometries
        soma_geom = CompartmentGeometry.sphere(diameter=20.0)  # 20 μm
        basal_geom = CompartmentGeometry.cylinder(length=150.0, diameter=2.0)
        apical_geom = CompartmentGeometry.cylinder(length=300.0, diameter=2.5)

        # Create channel populations
        if include_channels:
            soma_channels = ChannelPopulation({
                'Na': SodiumChannel(g_max=120.0, dt=dt),
                'K': PotassiumChannel(g_max=36.0, dt=dt),
                'Ca': CalciumChannel(g_max=5.0, dt=dt),
            })

            basal_channels = ChannelPopulation({
                'Na': SodiumChannel(g_max=60.0, dt=dt),
                'K': PotassiumChannel(g_max=18.0, dt=dt),
                'KA': ATypeKChannel(g_max=20.0, dt=dt),
            })

            apical_channels = ChannelPopulation({
                'Na': SodiumChannel(g_max=60.0, dt=dt),
                'K': PotassiumChannel(g_max=18.0, dt=dt),
                'KA': ATypeKChannel(g_max=30.0, dt=dt),
                'Ca': CalciumChannel(g_max=10.0, dt=dt),
                'HCN': HCNChannel(g_max=0.5, dt=dt),
            })
        else:
            # Passive only
            soma_channels = ChannelPopulation({})
            basal_channels = ChannelPopulation({})
            apical_channels = ChannelPopulation({})

        # Create compartments
        soma = Compartment(soma_geom, soma_channels, cable_props, dt=dt)
        basal = Compartment(basal_geom, basal_channels, cable_props, dt=dt)
        apical = Compartment(apical_geom, apical_channels, cable_props, dt=dt)

        # Connect compartments
        g_soma_basal = PrefabNeurons.compute_coupling_conductance(
            soma, basal, cable_props
        )
        g_soma_apical = PrefabNeurons.compute_coupling_conductance(
            soma, apical, cable_props
        )

        basal.add_connection(soma, g_soma_basal)
        apical.add_connection(soma, g_soma_apical)

        return MultiCompartmentNeuron([soma, basal, apical], dt=dt)

    @staticmethod
    def interneuron(
        include_channels: bool = True,
        dt: float = 0.01
    ) -> MultiCompartmentNeuron:
        """
        Create 2-compartment interneuron (fast-spiking).

        Compartments:
        1. Soma
        2. Dendrite

        Args:
            include_channels: Whether to include active channels
            dt: Time step (ms)

        Returns:
            Interneuron model
        """
        from .ion_channels import SodiumChannel, PotassiumChannel

        cable_props = CableProperties()

        # Smaller, faster interneuron
        soma_geom = CompartmentGeometry.sphere(diameter=15.0)
        dend_geom = CompartmentGeometry.cylinder(length=100.0, diameter=1.5)

        if include_channels:
            # Higher Na/K for fast spiking
            soma_channels = ChannelPopulation({
                'Na': SodiumChannel(g_max=150.0, dt=dt),
                'K': PotassiumChannel(g_max=50.0, dt=dt),
            })

            dend_channels = ChannelPopulation({
                'Na': SodiumChannel(g_max=75.0, dt=dt),
                'K': PotassiumChannel(g_max=25.0, dt=dt),
            })
        else:
            soma_channels = ChannelPopulation({})
            dend_channels = ChannelPopulation({})

        soma = Compartment(soma_geom, soma_channels, cable_props, dt=dt)
        dendrite = Compartment(dend_geom, dend_channels, cable_props, dt=dt)

        g_couple = PrefabNeurons.compute_coupling_conductance(
            soma, dendrite, cable_props
        )
        dendrite.add_connection(soma, g_couple)

        return MultiCompartmentNeuron([soma, dendrite], dt=dt)


class DendriticComputationAnalyzer:
    """
    Analyze dendritic computation in multi-compartment neurons.

    Quantifies:
    - Input integration (linear vs. nonlinear)
    - Dendritic spikes
    - Coincidence detection
    - Compartmentalization
    """

    def __init__(self, neuron: MultiCompartmentNeuron):
        self.neuron = neuron

    def measure_integration_mode(
        self,
        I_syn_dend: List[torch.Tensor],
        n_inputs: int = 2
    ) -> Dict[str, float]:
        """
        Measure whether dendrites integrate inputs linearly or nonlinearly.

        Args:
            I_syn_dend: Dendritic synaptic inputs
            n_inputs: Number of input combinations

        Returns:
            Integration metrics
        """
        # Measure individual responses
        responses_individual = []
        for i in range(n_inputs):
            inputs = [torch.zeros_like(I_syn_dend[i]) for _ in range(len(self.neuron.compartments))]
            inputs[1] = I_syn_dend[i]  # Apply to dendrite

            self.neuron.reset()
            V = self.neuron.forward(inputs)
            responses_individual.append(V[:, -1].mean().item())

        # Measure combined response
        inputs_combined = [torch.zeros_like(I_syn_dend[0]) for _ in range(len(self.neuron.compartments))]
        for i in range(n_inputs):
            inputs_combined[1] = inputs_combined[1] + I_syn_dend[i]

        self.neuron.reset()
        V_combined = self.neuron.forward(inputs_combined)
        response_combined = V_combined[:, -1].mean().item()

        # Compute linearity index
        expected_linear = sum(responses_individual)
        nonlinearity_index = (response_combined - expected_linear) / expected_linear

        return {
            'individual_responses': responses_individual,
            'combined_response': response_combined,
            'expected_linear': expected_linear,
            'nonlinearity_index': nonlinearity_index
        }

    def detect_dendritic_spikes(
        self,
        V_trace: torch.Tensor,
        threshold: float = -20.0
    ) -> Tuple[int, List[int]]:
        """
        Detect dendritic spikes in voltage trace.

        Args:
            V_trace: Voltage trace (n_steps,)
            threshold: Spike threshold (mV)

        Returns:
            (n_spikes, spike_times)
        """
        above_threshold = V_trace > threshold
        spike_indices = torch.where(
            above_threshold[1:] & ~above_threshold[:-1]
        )[0].tolist()

        return len(spike_indices), spike_indices
