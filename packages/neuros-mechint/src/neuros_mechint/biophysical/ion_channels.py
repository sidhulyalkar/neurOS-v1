"""
Ion Channel Models

Biophysically realistic models of voltage-gated and ligand-gated ion channels
for mechanistic understanding of neural computation.

References:
- Hodgkin & Huxley (1952): Quantitative description of membrane current
- Destexhe et al. (1994): Kinetic models of synaptic transmission
- Bean (2007): The action potential in mammalian central neurons
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass


@dataclass
class ChannelKinetics:
    """Kinetic parameters for ion channel gating."""
    alpha: torch.Tensor  # Forward rate
    beta: torch.Tensor   # Backward rate
    tau: Optional[torch.Tensor] = None  # Time constant
    m_inf: Optional[torch.Tensor] = None  # Steady-state activation


class VoltageGatedChannel(nn.Module):
    """
    Base class for voltage-gated ion channels.

    Implements Hodgkin-Huxley formalism with arbitrary gating variables.
    """

    def __init__(
        self,
        g_max: float,  # Maximum conductance (mS/cm^2)
        E_rev: float,  # Reversal potential (mV)
        m_power: int = 1,  # Activation power
        h_power: int = 0,  # Inactivation power
        temperature: float = 36.0,  # Temperature (°C)
        dt: float = 0.01  # Time step (ms)
    ):
        super().__init__()
        self.g_max = g_max
        self.E_rev = E_rev
        self.m_power = m_power
        self.h_power = h_power

        # Temperature correction factor (Q10 = 3)
        self.phi = 3.0 ** ((temperature - 6.3) / 10.0)
        self.dt = dt

        # Gating variables (initialized in forward pass)
        self.register_buffer('m', None)
        self.register_buffer('h', None)

    def alpha_m(self, V: torch.Tensor) -> torch.Tensor:
        """Activation rate. Override in subclasses."""
        raise NotImplementedError

    def beta_m(self, V: torch.Tensor) -> torch.Tensor:
        """Deactivation rate. Override in subclasses."""
        raise NotImplementedError

    def alpha_h(self, V: torch.Tensor) -> torch.Tensor:
        """Inactivation rate. Override in subclasses."""
        raise NotImplementedError

    def beta_h(self, V: torch.Tensor) -> torch.Tensor:
        """De-inactivation rate. Override in subclasses."""
        raise NotImplementedError

    def update_gates(self, V: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update gating variables using exponential Euler method.

        Args:
            V: Membrane potential (mV)

        Returns:
            (m, h): Updated gating variables
        """
        # Initialize if needed
        if self.m is None:
            self.m = torch.zeros_like(V)
            self.h = torch.ones_like(V)

        # Activation
        if self.m_power > 0:
            am = self.alpha_m(V)
            bm = self.beta_m(V)
            tau_m = 1.0 / (am + bm) / self.phi
            m_inf = am / (am + bm)
            self.m = m_inf - (m_inf - self.m) * torch.exp(-self.dt / tau_m)

        # Inactivation
        if self.h_power > 0:
            ah = self.alpha_h(V)
            bh = self.beta_h(V)
            tau_h = 1.0 / (ah + bh) / self.phi
            h_inf = ah / (ah + bh)
            self.h = h_inf - (h_inf - self.h) * torch.exp(-self.dt / tau_h)

        return self.m, self.h

    def current(self, V: torch.Tensor) -> torch.Tensor:
        """
        Compute channel current.

        Args:
            V: Membrane potential (mV)

        Returns:
            I: Channel current (μA/cm^2)
        """
        m, h = self.update_gates(V)

        g = self.g_max * (m ** self.m_power)
        if self.h_power > 0:
            g *= h ** self.h_power

        I = g * (V - self.E_rev)
        return I

    def conductance(self) -> torch.Tensor:
        """Get current conductance."""
        g = self.g_max * (self.m ** self.m_power)
        if self.h_power > 0:
            g *= self.h ** self.h_power
        return g

    def reset(self):
        """Reset gating variables."""
        self.m = None
        self.h = None


class SodiumChannel(VoltageGatedChannel):
    """
    Fast sodium channel (Na_v).

    Responsible for action potential upstroke.
    Based on Hodgkin-Huxley formulation.
    """

    def __init__(
        self,
        g_max: float = 120.0,  # mS/cm^2
        E_rev: float = 50.0,   # mV
        **kwargs
    ):
        super().__init__(g_max, E_rev, m_power=3, h_power=1, **kwargs)

    def alpha_m(self, V: torch.Tensor) -> torch.Tensor:
        """Sodium activation rate."""
        return 0.1 * (V + 40.0) / (1.0 - torch.exp(-(V + 40.0) / 10.0))

    def beta_m(self, V: torch.Tensor) -> torch.Tensor:
        """Sodium deactivation rate."""
        return 4.0 * torch.exp(-(V + 65.0) / 18.0)

    def alpha_h(self, V: torch.Tensor) -> torch.Tensor:
        """Sodium inactivation rate."""
        return 0.07 * torch.exp(-(V + 65.0) / 20.0)

    def beta_h(self, V: torch.Tensor) -> torch.Tensor:
        """Sodium de-inactivation rate."""
        return 1.0 / (1.0 + torch.exp(-(V + 35.0) / 10.0))


class PotassiumChannel(VoltageGatedChannel):
    """
    Delayed rectifier potassium channel (K_v).

    Responsible for action potential repolarization.
    Based on Hodgkin-Huxley formulation.
    """

    def __init__(
        self,
        g_max: float = 36.0,   # mS/cm^2
        E_rev: float = -77.0,  # mV
        **kwargs
    ):
        super().__init__(g_max, E_rev, m_power=4, h_power=0, **kwargs)

    def alpha_m(self, V: torch.Tensor) -> torch.Tensor:
        """Potassium activation rate."""
        return 0.01 * (V + 55.0) / (1.0 - torch.exp(-(V + 55.0) / 10.0))

    def beta_m(self, V: torch.Tensor) -> torch.Tensor:
        """Potassium deactivation rate."""
        return 0.125 * torch.exp(-(V + 65.0) / 80.0)

    def alpha_h(self, V: torch.Tensor) -> torch.Tensor:
        """Not used for delayed rectifier."""
        return torch.zeros_like(V)

    def beta_h(self, V: torch.Tensor) -> torch.Tensor:
        """Not used for delayed rectifier."""
        return torch.zeros_like(V)


class ATypeKChannel(VoltageGatedChannel):
    """
    A-type potassium channel (K_A).

    Fast transient potassium channel involved in action potential
    timing and dendritic computation.
    """

    def __init__(
        self,
        g_max: float = 20.0,   # mS/cm^2
        E_rev: float = -77.0,  # mV
        **kwargs
    ):
        super().__init__(g_max, E_rev, m_power=3, h_power=1, **kwargs)

    def alpha_m(self, V: torch.Tensor) -> torch.Tensor:
        """A-type activation rate."""
        return 0.02 * (V + 55.0) / (1.0 - torch.exp(-(V + 55.0) / 11.0))

    def beta_m(self, V: torch.Tensor) -> torch.Tensor:
        """A-type deactivation rate."""
        return 0.5 * torch.exp(-(V + 60.0) / 80.0)

    def alpha_h(self, V: torch.Tensor) -> torch.Tensor:
        """A-type inactivation rate."""
        return 0.015 * torch.exp(-(V + 60.0) / 20.0)

    def beta_h(self, V: torch.Tensor) -> torch.Tensor:
        """A-type de-inactivation rate."""
        return 0.5 / (1.0 + torch.exp(-(V + 50.0) / 15.0))


class CalciumChannel(VoltageGatedChannel):
    """
    High-voltage activated calcium channel (Ca_v).

    Mediates calcium influx for synaptic plasticity and
    intracellular signaling cascades.
    """

    def __init__(
        self,
        g_max: float = 10.0,   # mS/cm^2
        E_rev: float = 120.0,  # mV
        **kwargs
    ):
        super().__init__(g_max, E_rev, m_power=2, h_power=1, **kwargs)

    def alpha_m(self, V: torch.Tensor) -> torch.Tensor:
        """Calcium activation rate."""
        return 1.6 / (1.0 + torch.exp(-0.072 * (V - 5.0)))

    def beta_m(self, V: torch.Tensor) -> torch.Tensor:
        """Calcium deactivation rate."""
        return 0.02 * (V + 8.9) / (torch.exp((V + 8.9) / 5.0) - 1.0)

    def alpha_h(self, V: torch.Tensor) -> torch.Tensor:
        """Calcium inactivation rate."""
        return 0.0197 * torch.exp(-(V + 25.0) / 20.0)

    def beta_h(self, V: torch.Tensor) -> torch.Tensor:
        """Calcium de-inactivation rate."""
        return 0.5 / (1.0 + torch.exp(-(V + 40.0) / 5.0))


class HCNChannel(VoltageGatedChannel):
    """
    Hyperpolarization-activated cyclic nucleotide-gated (HCN) channel.

    Mediates I_h current, important for rhythmogenesis and
    dendritic integration.
    """

    def __init__(
        self,
        g_max: float = 0.5,    # mS/cm^2
        E_rev: float = -45.0,  # mV (non-selective cation)
        **kwargs
    ):
        super().__init__(g_max, E_rev, m_power=1, h_power=0, **kwargs)

    def alpha_m(self, V: torch.Tensor) -> torch.Tensor:
        """HCN activation rate (activated by hyperpolarization)."""
        return 0.001 * torch.exp(-(V + 75.0) / 5.5)

    def beta_m(self, V: torch.Tensor) -> torch.Tensor:
        """HCN deactivation rate."""
        return 0.001 * torch.exp((V + 75.0) / 5.5)

    def alpha_h(self, V: torch.Tensor) -> torch.Tensor:
        """Not used."""
        return torch.zeros_like(V)

    def beta_h(self, V: torch.Tensor) -> torch.Tensor:
        """Not used."""
        return torch.zeros_like(V)


class LigandGatedChannel(nn.Module):
    """
    Base class for ligand-gated ion channels (receptors).

    Implements kinetic schemes for neurotransmitter binding.
    """

    def __init__(
        self,
        g_max: float,
        E_rev: float,
        alpha: float,  # Binding rate
        beta: float,   # Unbinding rate
        dt: float = 0.01
    ):
        super().__init__()
        self.g_max = g_max
        self.E_rev = E_rev
        self.alpha = alpha
        self.beta = beta
        self.dt = dt

        # Fraction of open channels
        self.register_buffer('O', None)

    def update(self, NT: torch.Tensor) -> torch.Tensor:
        """
        Update open probability based on neurotransmitter concentration.

        Args:
            NT: Neurotransmitter concentration (mM)

        Returns:
            O: Open probability
        """
        if self.O is None:
            self.O = torch.zeros_like(NT)

        # dO/dt = alpha * NT * (1 - O) - beta * O
        dO = self.alpha * NT * (1.0 - self.O) - self.beta * self.O
        self.O = self.O + self.dt * dO
        self.O = torch.clamp(self.O, 0.0, 1.0)

        return self.O

    def current(self, V: torch.Tensor, NT: torch.Tensor) -> torch.Tensor:
        """
        Compute synaptic current.

        Args:
            V: Postsynaptic membrane potential (mV)
            NT: Neurotransmitter concentration (mM)

        Returns:
            I: Synaptic current (μA/cm^2)
        """
        O = self.update(NT)
        g = self.g_max * O
        I = g * (V - self.E_rev)
        return I

    def reset(self):
        """Reset channel state."""
        self.O = None


class AMPAReceptor(LigandGatedChannel):
    """
    AMPA glutamate receptor.

    Fast excitatory synaptic transmission.
    """

    def __init__(
        self,
        g_max: float = 1.0,   # nS
        E_rev: float = 0.0,   # mV
        alpha: float = 1.1,   # /mM /ms
        beta: float = 0.19,   # /ms
        **kwargs
    ):
        super().__init__(g_max, E_rev, alpha, beta, **kwargs)


class NMDAReceptor(nn.Module):
    """
    NMDA glutamate receptor.

    Voltage-dependent slow excitatory transmission with
    magnesium block.
    """

    def __init__(
        self,
        g_max: float = 1.0,   # nS
        E_rev: float = 0.0,   # mV
        alpha: float = 0.072, # /mM /ms
        beta: float = 0.0066, # /ms
        Mg: float = 1.0,      # mM (external magnesium)
        dt: float = 0.01
    ):
        super().__init__()
        self.g_max = g_max
        self.E_rev = E_rev
        self.alpha = alpha
        self.beta = beta
        self.Mg = Mg
        self.dt = dt

        self.register_buffer('O', None)

    def mg_block(self, V: torch.Tensor) -> torch.Tensor:
        """
        Voltage-dependent magnesium block.

        Args:
            V: Membrane potential (mV)

        Returns:
            B: Fraction of unblocked channels [0, 1]
        """
        return 1.0 / (1.0 + self.Mg * torch.exp(-0.062 * V) / 3.57)

    def update(self, NT: torch.Tensor) -> torch.Tensor:
        """Update open probability."""
        if self.O is None:
            self.O = torch.zeros_like(NT)

        dO = self.alpha * NT * (1.0 - self.O) - self.beta * self.O
        self.O = self.O + self.dt * dO
        self.O = torch.clamp(self.O, 0.0, 1.0)

        return self.O

    def current(self, V: torch.Tensor, NT: torch.Tensor) -> torch.Tensor:
        """
        Compute NMDA current with Mg2+ block.

        Args:
            V: Membrane potential (mV)
            NT: Glutamate concentration (mM)

        Returns:
            I: NMDA current (μA/cm^2)
        """
        O = self.update(NT)
        B = self.mg_block(V)
        g = self.g_max * O * B
        I = g * (V - self.E_rev)
        return I

    def reset(self):
        """Reset receptor state."""
        self.O = None


class GABAAReceptor(LigandGatedChannel):
    """
    GABA-A receptor.

    Fast inhibitory synaptic transmission.
    """

    def __init__(
        self,
        g_max: float = 1.0,   # nS
        E_rev: float = -70.0, # mV (chloride reversal)
        alpha: float = 5.0,   # /mM /ms
        beta: float = 0.18,   # /ms
        **kwargs
    ):
        super().__init__(g_max, E_rev, alpha, beta, **kwargs)


class GABABReceptor(nn.Module):
    """
    GABA-B receptor.

    Slow metabotropic inhibitory transmission via G-proteins
    and GIRK channels.
    """

    def __init__(
        self,
        g_max: float = 0.5,   # nS
        E_rev: float = -95.0, # mV (potassium reversal)
        K1: float = 0.09,     # /ms
        K2: float = 0.0012,   # /ms
        K3: float = 0.18,     # /ms
        K4: float = 0.034,    # /ms
        dt: float = 0.01
    ):
        super().__init__()
        self.g_max = g_max
        self.E_rev = E_rev
        self.K1 = K1
        self.K2 = K2
        self.K3 = K3
        self.K4 = K4
        self.dt = dt

        # G-protein cascade states
        self.register_buffer('R', None)  # Receptor
        self.register_buffer('G', None)  # G-protein

    def update(self, NT: torch.Tensor) -> torch.Tensor:
        """
        Update G-protein cascade.

        Args:
            NT: GABA concentration (mM)

        Returns:
            G: Active G-protein fraction
        """
        if self.R is None:
            self.R = torch.zeros_like(NT)
            self.G = torch.zeros_like(NT)

        # Receptor binding
        dR = self.K1 * NT * (1.0 - self.R) - self.K2 * self.R
        self.R = self.R + self.dt * dR
        self.R = torch.clamp(self.R, 0.0, 1.0)

        # G-protein activation
        dG = self.K3 * self.R - self.K4 * self.G
        self.G = self.G + self.dt * dG
        self.G = torch.clamp(self.G, 0.0, 1.0)

        return self.G

    def current(self, V: torch.Tensor, NT: torch.Tensor) -> torch.Tensor:
        """
        Compute GABA-B current.

        Args:
            V: Membrane potential (mV)
            NT: GABA concentration (mM)

        Returns:
            I: GABA-B current (μA/cm^2)
        """
        G = self.update(NT)
        # Quadratic G-protein dependence
        g = self.g_max * G * G / (G * G + 0.01)
        I = g * (V - self.E_rev)
        return I

    def reset(self):
        """Reset receptor state."""
        self.R = None
        self.G = None


class ChannelPopulation(nn.Module):
    """
    Collection of ion channels for a neuron compartment.

    Manages multiple channel types and computes total membrane current.
    """

    def __init__(self, channels: Dict[str, nn.Module]):
        super().__init__()
        self.channels = nn.ModuleDict(channels)

    def total_current(
        self,
        V: torch.Tensor,
        NT: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Compute total membrane current from all channels.

        Args:
            V: Membrane potential (mV)
            NT: Dictionary of neurotransmitter concentrations (optional)

        Returns:
            I_total: Total current (μA/cm^2)
        """
        I_total = torch.zeros_like(V)

        for name, channel in self.channels.items():
            if isinstance(channel, VoltageGatedChannel):
                I_total += channel.current(V)
            elif isinstance(channel, (LigandGatedChannel, NMDAReceptor, GABABReceptor)):
                if NT is not None and name in NT:
                    I_total += channel.current(V, NT[name])

        return I_total

    def get_conductances(self) -> Dict[str, torch.Tensor]:
        """Get conductances for all voltage-gated channels."""
        conductances = {}
        for name, channel in self.channels.items():
            if isinstance(channel, VoltageGatedChannel):
                conductances[name] = channel.conductance()
        return conductances

    def reset_all(self):
        """Reset all channel states."""
        for channel in self.channels.values():
            channel.reset()
