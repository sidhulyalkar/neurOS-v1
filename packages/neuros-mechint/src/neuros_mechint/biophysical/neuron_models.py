"""
Extended Neuron Models

Collection of biophysically-detailed neuron models beyond basic LIF.

References:
- Izhikevich (2003): Simple model of spiking neurons
- Brette & Gerstner (2005): Adaptive exponential integrate-and-fire
- Pinsky & Rinzel (1994): Intrinsic and network rhythmogenesis
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict
from dataclasses import dataclass


@dataclass
class NeuronParameters:
    """Parameters for neuron models."""
    C: float = 1.0    # Membrane capacitance (μF/cm^2)
    g_L: float = 0.1  # Leak conductance (mS/cm^2)
    E_L: float = -65.0  # Leak reversal potential (mV)
    V_th: float = -50.0  # Spike threshold (mV)
    V_reset: float = -65.0  # Reset potential (mV)
    dt: float = 0.1  # Time step (ms)


class BiophysicalNeuronBase(nn.Module):
    """Base class for biophysical neuron models."""

    def __init__(self, params: NeuronParameters):
        super().__init__()
        self.params = params

    def reset_state(self):
        """Reset neuron state. Override in subclasses."""
        raise NotImplementedError

    def forward(self, I: torch.Tensor, n_steps: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Simulate neuron.

        Args:
            I: Input current (μA/cm^2) of shape (batch, n_steps)
            n_steps: Number of time steps

        Returns:
            (V, spikes): Voltage trace and spike train
        """
        raise NotImplementedError


class AdExNeuron(BiophysicalNeuronBase):
    """
    Adaptive Exponential Integrate-and-Fire (AdEx) neuron.

    Captures spike initiation dynamics and adaptation.

    dV/dt = (g_L * (E_L - V) + g_L * Δ_T * exp((V - V_T) / Δ_T) + I - w) / C
    dw/dt = (a * (V - E_L) - w) / τ_w

    References:
        Brette & Gerstner (2005), J. Neurophysiol.
    """

    def __init__(
        self,
        params: Optional[NeuronParameters] = None,
        Delta_T: float = 2.0,  # Spike slope factor (mV)
        V_T: float = -50.0,    # Threshold (mV)
        a: float = 0.001,      # Subthreshold adaptation (μS)
        b: float = 0.1,        # Spike-triggered adaptation (nA)
        tau_w: float = 100.0,  # Adaptation time constant (ms)
        V_peak: float = 20.0   # Peak voltage (mV)
    ):
        if params is None:
            params = NeuronParameters()
        super().__init__(params)

        self.Delta_T = Delta_T
        self.V_T = V_T
        self.a = a
        self.b = b
        self.tau_w = tau_w
        self.V_peak = V_peak

        # State variables
        self.register_buffer('V', None)
        self.register_buffer('w', None)

    def reset_state(self):
        """Reset to resting state."""
        if self.V is not None:
            self.V.fill_(self.params.E_L)
            self.w.fill_(0.0)

    def forward(
        self,
        I: torch.Tensor,
        n_steps: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Simulate AdEx neuron.

        Args:
            I: Input current (batch, n_steps)
            n_steps: Number of time steps

        Returns:
            (V_trace, spike_train)
        """
        batch_size = I.shape[0]
        device = I.device

        # Initialize state
        if self.V is None or self.V.shape[0] != batch_size:
            self.V = torch.full((batch_size,), self.params.E_L, device=device)
            self.w = torch.zeros(batch_size, device=device)

        V_trace = torch.zeros(batch_size, n_steps, device=device)
        spike_train = torch.zeros(batch_size, n_steps, device=device)

        dt = self.params.dt
        C = self.params.C
        g_L = self.params.g_L
        E_L = self.params.E_L

        for t in range(n_steps):
            # Exponential spike current
            exp_term = g_L * self.Delta_T * torch.exp((self.V - self.V_T) / self.Delta_T)

            # Voltage dynamics
            dV = (g_L * (E_L - self.V) + exp_term + I[:, t] - self.w) / C
            self.V = self.V + dt * dV

            # Adaptation dynamics
            dw = (self.a * (self.V - E_L) - self.w) / self.tau_w
            self.w = self.w + dt * dw

            # Spike detection and reset
            spiked = self.V >= self.V_peak
            spike_train[:, t] = spiked.float()

            self.V = torch.where(spiked, torch.tensor(self.params.V_reset, device=device), self.V)
            self.w = torch.where(spiked, self.w + self.b, self.w)

            V_trace[:, t] = self.V

        return V_trace, spike_train


class QuadraticIFNeuron(BiophysicalNeuronBase):
    """
    Quadratic Integrate-and-Fire (QIF) neuron.

    Normal form of Type I excitability.

    dV/dt = (a * (V - V_rest) * (V - V_c) + I) / C

    References:
        Ermentrout & Kopell (1986), SIAM J. Appl. Math.
    """

    def __init__(
        self,
        params: Optional[NeuronParameters] = None,
        a: float = 0.04,       # Curvature parameter
        V_rest: float = -65.0, # Resting potential (mV)
        V_c: float = -55.0,    # Critical voltage (mV)
        V_peak: float = 20.0   # Peak voltage (mV)
    ):
        if params is None:
            params = NeuronParameters()
        super().__init__(params)

        self.a = a
        self.V_rest = V_rest
        self.V_c = V_c
        self.V_peak = V_peak

        self.register_buffer('V', None)

    def reset_state(self):
        """Reset to resting state."""
        if self.V is not None:
            self.V.fill_(self.V_rest)

    def forward(
        self,
        I: torch.Tensor,
        n_steps: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Simulate QIF neuron.

        Args:
            I: Input current (batch, n_steps)
            n_steps: Number of time steps

        Returns:
            (V_trace, spike_train)
        """
        batch_size = I.shape[0]
        device = I.device

        if self.V is None or self.V.shape[0] != batch_size:
            self.V = torch.full((batch_size,), self.V_rest, device=device)

        V_trace = torch.zeros(batch_size, n_steps, device=device)
        spike_train = torch.zeros(batch_size, n_steps, device=device)

        dt = self.params.dt
        C = self.params.C

        for t in range(n_steps):
            # Quadratic nonlinearity
            dV = (self.a * (self.V - self.V_rest) * (self.V - self.V_c) + I[:, t]) / C
            self.V = self.V + dt * dV

            # Spike detection and reset
            spiked = self.V >= self.V_peak
            spike_train[:, t] = spiked.float()
            self.V = torch.where(spiked, torch.tensor(self.params.V_reset, device=device), self.V)

            V_trace[:, t] = self.V

        return V_trace, spike_train


class ResonateAndFireNeuron(BiophysicalNeuronBase):
    """
    Resonate-and-Fire neuron.

    Models subthreshold oscillations and resonance.

    dV/dt = (I - g_L * (V - E_L) - w) / C
    dw/dt = (V - w) / τ

    References:
        Izhikevich (2001), IEEE Trans. Neural Netw.
    """

    def __init__(
        self,
        params: Optional[NeuronParameters] = None,
        tau: float = 10.0,  # Resonance time constant (ms)
        V_peak: float = 30.0
    ):
        if params is None:
            params = NeuronParameters()
        super().__init__(params)

        self.tau = tau
        self.V_peak = V_peak

        self.register_buffer('V', None)
        self.register_buffer('w', None)

    def reset_state(self):
        """Reset state."""
        if self.V is not None:
            self.V.fill_(self.params.E_L)
            self.w.fill_(0.0)

    def forward(
        self,
        I: torch.Tensor,
        n_steps: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Simulate resonate-and-fire neuron."""
        batch_size = I.shape[0]
        device = I.device

        if self.V is None or self.V.shape[0] != batch_size:
            self.V = torch.full((batch_size,), self.params.E_L, device=device)
            self.w = torch.zeros(batch_size, device=device)

        V_trace = torch.zeros(batch_size, n_steps, device=device)
        spike_train = torch.zeros(batch_size, n_steps, device=device)

        dt = self.params.dt
        C = self.params.C
        g_L = self.params.g_L
        E_L = self.params.E_L

        for t in range(n_steps):
            dV = (I[:, t] - g_L * (self.V - E_L) - self.w) / C
            dw = (self.V - self.w) / self.tau

            self.V = self.V + dt * dV
            self.w = self.w + dt * dw

            spiked = self.V >= self.V_peak
            spike_train[:, t] = spiked.float()
            self.V = torch.where(spiked, torch.tensor(self.params.V_reset, device=device), self.V)

            V_trace[:, t] = self.V

        return V_trace, spike_train


class PinskyRinzelNeuron(nn.Module):
    """
    Two-compartment Pinsky-Rinzel model.

    Simplified model of CA3 pyramidal cell with soma and dendrite.

    References:
        Pinsky & Rinzel (1994), J. Comp. Neurosci.
    """

    def __init__(
        self,
        g_Na: float = 30.0,   # Somatic Na conductance
        g_K_DR: float = 15.0,  # Somatic K conductance
        g_Ca: float = 10.0,    # Dendritic Ca conductance
        g_K_AHP: float = 0.8,  # Dendritic K(Ca) conductance
        g_K_C: float = 15.0,   # Dendritic K(C) conductance
        g_c: float = 2.1,      # Coupling conductance
        dt: float = 0.01
    ):
        super().__init__()
        self.g_Na = g_Na
        self.g_K_DR = g_K_DR
        self.g_Ca = g_Ca
        self.g_K_AHP = g_K_AHP
        self.g_K_C = g_K_C
        self.g_c = g_c
        self.dt = dt

        # Reversal potentials
        self.E_Na = 60.0
        self.E_K = -75.0
        self.E_Ca = 80.0
        self.E_L = -60.0

        # Capacitances
        self.C_s = 1.0  # Soma
        self.C_d = 1.0  # Dendrite

        # State variables
        self.register_buffer('V_s', None)  # Somatic voltage
        self.register_buffer('V_d', None)  # Dendritic voltage
        self.register_buffer('Ca', None)   # Calcium concentration

        # Gating variables
        self.register_buffer('h', None)    # Na inactivation
        self.register_buffer('n', None)    # K activation
        self.register_buffer('s', None)    # Ca activation
        self.register_buffer('c', None)    # K(Ca) activation
        self.register_buffer('q', None)    # K(C) activation

    def reset_state(self):
        """Reset to resting state."""
        if self.V_s is not None:
            self.V_s.fill_(-60.0)
            self.V_d.fill_(-60.0)
            self.Ca.fill_(0.1)
            self.h.fill_(0.9)
            self.n.fill_(0.0)
            self.s.fill_(0.0)
            self.c.fill_(0.0)
            self.q.fill_(0.0)

    def forward(
        self,
        I_s: torch.Tensor,
        I_d: torch.Tensor,
        n_steps: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Simulate Pinsky-Rinzel neuron.

        Args:
            I_s: Somatic current (batch, n_steps)
            I_d: Dendritic current (batch, n_steps)
            n_steps: Number of time steps

        Returns:
            (V_s_trace, V_d_trace): Somatic and dendritic voltage traces
        """
        batch_size = I_s.shape[0]
        device = I_s.device

        # Initialize
        if self.V_s is None or self.V_s.shape[0] != batch_size:
            self.V_s = torch.full((batch_size,), -60.0, device=device)
            self.V_d = torch.full((batch_size,), -60.0, device=device)
            self.Ca = torch.full((batch_size,), 0.1, device=device)
            self.h = torch.full((batch_size,), 0.9, device=device)
            self.n = torch.full((batch_size,), 0.0, device=device)
            self.s = torch.full((batch_size,), 0.0, device=device)
            self.c = torch.full((batch_size,), 0.0, device=device)
            self.q = torch.full((batch_size,), 0.0, device=device)

        V_s_trace = torch.zeros(batch_size, n_steps, device=device)
        V_d_trace = torch.zeros(batch_size, n_steps, device=device)

        for t in range(n_steps):
            # Soma currents
            m_inf = 1.0 / (1.0 + torch.exp(-(self.V_s + 37.0) / 10.0))
            I_Na = self.g_Na * (m_inf ** 2) * self.h * (self.V_s - self.E_Na)
            I_K_DR = self.g_K_DR * self.n * (self.V_s - self.E_K)
            I_L_s = 0.1 * (self.V_s - self.E_L)

            # Dendrite currents
            s_inf = 1.0 / (1.0 + torch.exp(-(self.V_d + 10.0) / 7.0))
            I_Ca = self.g_Ca * s_inf * self.s * (self.V_d - self.E_Ca)
            I_K_AHP = self.g_K_AHP * self.c * (self.V_d - self.E_K)
            I_K_C = self.g_K_C * torch.min(self.Ca / 250.0, torch.ones_like(self.Ca)) * self.q * (self.V_d - self.E_K)
            I_L_d = 0.1 * (self.V_d - self.E_L)

            # Coupling current
            I_couple = self.g_c * (self.V_d - self.V_s)

            # Voltage updates
            dV_s = (-I_Na - I_K_DR - I_L_s + I_couple + I_s[:, t]) / self.C_s
            dV_d = (-I_Ca - I_K_AHP - I_K_C - I_L_d - I_couple + I_d[:, t]) / self.C_d

            self.V_s = self.V_s + self.dt * dV_s
            self.V_d = self.V_d + self.dt * dV_d

            # Gating variable updates (simplified)
            h_inf = 1.0 / (1.0 + torch.exp((self.V_s + 41.0) / 4.0))
            n_inf = 1.0 / (1.0 + torch.exp(-(self.V_s + 34.0) / 10.0))
            c_inf = torch.min(self.Ca / 250.0, torch.ones_like(self.Ca))
            q_inf = torch.min(self.Ca / 500.0, torch.ones_like(self.Ca))

            self.h = h_inf - (h_inf - self.h) * torch.exp(-self.dt / 1.0)
            self.n = n_inf - (n_inf - self.n) * torch.exp(-self.dt / 5.0)
            self.s = s_inf - (s_inf - self.s) * torch.exp(-self.dt / 10.0)
            self.c = c_inf - (c_inf - self.c) * torch.exp(-self.dt / 80.0)
            self.q = q_inf - (q_inf - self.q) * torch.exp(-self.dt / 50.0)

            # Calcium dynamics
            dCa = -0.13 * I_Ca - 0.075 * self.Ca
            self.Ca = self.Ca + self.dt * dCa

            V_s_trace[:, t] = self.V_s
            V_d_trace[:, t] = self.V_d

        return V_s_trace, V_d_trace
