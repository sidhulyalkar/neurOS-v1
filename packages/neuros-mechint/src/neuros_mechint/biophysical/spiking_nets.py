"""
Differentiable Spiking Neural Networks

Biophysically-inspired spiking neuron models with surrogate gradients for backpropagation.
Supports LIF, Izhikevich, and Hodgkin-Huxley neurons.

Enables training SNNs end-to-end with standard gradient descent while maintaining
biological plausibility.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np
import logging

logger = logging.getLogger(__name__)


class SurrogateGradient(torch.autograd.Function):
    """
    Surrogate gradient for spiking neurons.

    The Heaviside step function (spike) has zero gradient almost everywhere,
    preventing backpropagation. We replace it with a smooth approximation
    for the backward pass.

    Forward: spike = H(v - threshold)
    Backward: dL/dv = dL/dspike * sigmoid_derivative(v - threshold)
    """

    @staticmethod
    def forward(ctx, input: Tensor, threshold: float = 1.0, beta: float = 1.0):
        """
        Forward pass: Heaviside step function.

        Args:
            input: Membrane potential
            threshold: Spike threshold
            beta: Surrogate gradient sharpness

        Returns:
            Binary spikes (0 or 1)
        """
        ctx.save_for_backward(input)
        ctx.threshold = threshold
        ctx.beta = beta

        return (input >= threshold).float()

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        """
        Backward pass: Smooth sigmoid gradient.

        Args:
            grad_output: Gradient from downstream

        Returns:
            Gradient w.r.t. input
        """
        input, = ctx.saved_tensors
        threshold = ctx.threshold
        beta = ctx.beta

        # Sigmoid derivative as surrogate
        # f'(x) = β * sigmoid(β*x) * (1 - sigmoid(β*x))
        z = beta * (input - threshold)
        sigmoid = torch.sigmoid(z)
        grad = beta * sigmoid * (1 - sigmoid)

        return grad_output * grad, None, None


class LeakyIntegrateFireNeuron(nn.Module):
    """
    Leaky Integrate-and-Fire (LIF) neuron model.

    Membrane dynamics:
        τ * dV/dt = -(V - V_rest) + I(t)

    If V ≥ θ: emit spike and reset V to V_reset

    Args:
        tau_mem: Membrane time constant (ms)
        v_threshold: Spike threshold
        v_reset: Reset potential after spike
        v_rest: Resting potential
        surrogate_beta: Surrogate gradient sharpness
        device: Torch device

    Example:
        >>> lif = LeakyIntegrateFireNeuron(tau_mem=10.0)
        >>> spikes, voltages = lif(input_current, dt=1.0, n_steps=100)
    """

    def __init__(
        self,
        tau_mem: float = 10.0,
        v_threshold: float = 1.0,
        v_reset: float = 0.0,
        v_rest: float = 0.0,
        surrogate_beta: float = 1.0,
        device: Optional[str] = None,
    ):
        super().__init__()

        self.tau_mem = tau_mem
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.v_rest = v_rest
        self.surrogate_beta = surrogate_beta
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(
        self,
        input_current: Tensor,
        dt: float = 1.0,
        v_init: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Simulate LIF neuron dynamics.

        Args:
            input_current: Input current [batch, n_steps] or [batch, n_steps, n_neurons]
            dt: Time step size (ms)
            v_init: Initial membrane potential

        Returns:
            spikes: Binary spike trains [batch, n_steps, ...]
            voltages: Membrane potentials [batch, n_steps, ...]
        """
        if input_current.dim() == 2:
            batch_size, n_steps = input_current.shape
            n_neurons = 1
            input_current = input_current.unsqueeze(2)
        else:
            batch_size, n_steps, n_neurons = input_current.shape

        # Initialize membrane potential
        if v_init is None:
            v = torch.ones(batch_size, n_neurons, device=self.device) * self.v_rest
        else:
            v = v_init.to(self.device)

        # Storage
        spikes_list = []
        voltages_list = []

        # Time loop
        for t in range(n_steps):
            # Membrane dynamics (Euler integration)
            # dV/dt = (-(V - V_rest) + I) / τ
            dv = (-(v - self.v_rest) + input_current[:, t, :]) / self.tau_mem
            v = v + dv * dt

            # Spike generation (using surrogate gradient)
            spike = SurrogateGradient.apply(v, self.v_threshold, self.surrogate_beta)

            # Reset after spike
            v = v * (1.0 - spike) + self.v_reset * spike

            spikes_list.append(spike)
            voltages_list.append(v)

        # Stack results
        spikes = torch.stack(spikes_list, dim=1)  # [batch, n_steps, n_neurons]
        voltages = torch.stack(voltages_list, dim=1)

        if n_neurons == 1:
            spikes = spikes.squeeze(2)
            voltages = voltages.squeeze(2)

        return spikes, voltages


class IzhikevichNeuron(nn.Module):
    """
    Izhikevich neuron model with rich dynamics.

    Provides diverse spiking behaviors (regular spiking, bursting, fast spiking, etc.)
    with only 2 state variables and 4 parameters.

    Dynamics:
        dv/dt = 0.04*v² + 5*v + 140 - u + I
        du/dt = a*(b*v - u)

    If v ≥ 30: emit spike, v ← c, u ← u + d

    Args:
        a: Recovery time scale (default: 0.02)
        b: Sensitivity of recovery (default: 0.2)
        c: Reset voltage (default: -65)
        d: Reset of recovery (default: 8)
        device: Torch device

    Example:
        >>> izh = IzhikevichNeuron(a=0.02, b=0.2)  # Regular spiking
        >>> spikes, voltages, recovery = izh(input_current, dt=0.5)
    """

    def __init__(
        self,
        a: float = 0.02,
        b: float = 0.2,
        c: float = -65.0,
        d: float = 8.0,
        v_threshold: float = 30.0,
        device: Optional[str] = None,
    ):
        super().__init__()

        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.v_threshold = v_threshold
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(
        self,
        input_current: Tensor,
        dt: float = 0.5,
        v_init: Optional[Tensor] = None,
        u_init: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Simulate Izhikevich neuron.

        Args:
            input_current: Input [batch, n_steps] or [batch, n_steps, n_neurons]
            dt: Time step (ms)
            v_init: Initial voltage
            u_init: Initial recovery variable

        Returns:
            spikes: Binary spikes
            voltages: Membrane potentials
            recovery: Recovery variables
        """
        if input_current.dim() == 2:
            batch_size, n_steps = input_current.shape
            n_neurons = 1
            input_current = input_current.unsqueeze(2)
        else:
            batch_size, n_steps, n_neurons = input_current.shape

        # Initialize
        if v_init is None:
            v = torch.ones(batch_size, n_neurons, device=self.device) * self.c
        else:
            v = v_init.to(self.device)

        if u_init is None:
            u = torch.ones(batch_size, n_neurons, device=self.device) * (self.b * self.c)
        else:
            u = u_init.to(self.device)

        spikes_list = []
        voltages_list = []
        recovery_list = []

        for t in range(n_steps):
            # Dynamics
            dv = (0.04 * v ** 2 + 5 * v + 140 - u + input_current[:, t, :]) * dt
            du = (self.a * (self.b * v - u)) * dt

            v = v + dv
            u = u + du

            # Spike detection
            spike = SurrogateGradient.apply(v, self.v_threshold, 1.0)

            # Reset
            v = v * (1.0 - spike) + self.c * spike
            u = u + self.d * spike

            spikes_list.append(spike)
            voltages_list.append(v)
            recovery_list.append(u)

        spikes = torch.stack(spikes_list, dim=1)
        voltages = torch.stack(voltages_list, dim=1)
        recovery = torch.stack(recovery_list, dim=1)

        if n_neurons == 1:
            spikes = spikes.squeeze(2)
            voltages = voltages.squeeze(2)
            recovery = recovery.squeeze(2)

        return spikes, voltages, recovery


class HodgkinHuxleyNeuron(nn.Module):
    """
    Hodgkin-Huxley conductance-based neuron model.

    Full biophysical model with sodium, potassium, and leak channels.
    Most realistic but computationally expensive.

    Dynamics:
        C * dV/dt = -I_Na - I_K - I_L + I_ext
        I_Na = g_Na * m³ * h * (V - E_Na)
        I_K = g_K * n⁴ * (V - E_K)
        I_L = g_L * (V - E_L)

    Args:
        g_na: Sodium conductance (default: 120 mS/cm²)
        g_k: Potassium conductance (default: 36 mS/cm²)
        g_l: Leak conductance (default: 0.3 mS/cm²)
        e_na: Sodium reversal potential (default: 50 mV)
        e_k: Potassium reversal potential (default: -77 mV)
        e_l: Leak reversal potential (default: -54.4 mV)
        c_m: Membrane capacitance (default: 1 μF/cm²)
        device: Torch device

    Example:
        >>> hh = HodgkinHuxleyNeuron()
        >>> spikes, voltages, gates = hh(input_current, dt=0.01)
    """

    def __init__(
        self,
        g_na: float = 120.0,
        g_k: float = 36.0,
        g_l: float = 0.3,
        e_na: float = 50.0,
        e_k: float = -77.0,
        e_l: float = -54.4,
        c_m: float = 1.0,
        device: Optional[str] = None,
    ):
        super().__init__()

        self.g_na = g_na
        self.g_k = g_k
        self.g_l = g_l
        self.e_na = e_na
        self.e_k = e_k
        self.e_l = e_l
        self.c_m = c_m
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    def _alpha_m(self, v: Tensor) -> Tensor:
        """Sodium activation rate."""
        return 0.1 * (v + 40.0) / (1.0 - torch.exp(-(v + 40.0) / 10.0))

    def _beta_m(self, v: Tensor) -> Tensor:
        """Sodium deactivation rate."""
        return 4.0 * torch.exp(-(v + 65.0) / 18.0)

    def _alpha_h(self, v: Tensor) -> Tensor:
        """Sodium inactivation rate."""
        return 0.07 * torch.exp(-(v + 65.0) / 20.0)

    def _beta_h(self, v: Tensor) -> Tensor:
        """Sodium deinactivation rate."""
        return 1.0 / (1.0 + torch.exp(-(v + 35.0) / 10.0))

    def _alpha_n(self, v: Tensor) -> Tensor:
        """Potassium activation rate."""
        return 0.01 * (v + 55.0) / (1.0 - torch.exp(-(v + 55.0) / 10.0))

    def _beta_n(self, v: Tensor) -> Tensor:
        """Potassium deactivation rate."""
        return 0.125 * torch.exp(-(v + 65.0) / 80.0)

    def forward(
        self,
        input_current: Tensor,
        dt: float = 0.01,
        v_init: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
        """
        Simulate Hodgkin-Huxley neuron.

        Args:
            input_current: Input current [batch, n_steps] or [batch, n_steps, n_neurons]
            dt: Time step (ms) - must be small (< 0.1) for stability
            v_init: Initial voltage

        Returns:
            spikes: Binary spikes (detected from voltage crossings)
            voltages: Membrane potentials
            gates: Dictionary with gating variables {m, h, n}
        """
        if input_current.dim() == 2:
            batch_size, n_steps = input_current.shape
            n_neurons = 1
            input_current = input_current.unsqueeze(2)
        else:
            batch_size, n_steps, n_neurons = input_current.shape

        # Initialize
        if v_init is None:
            v = torch.ones(batch_size, n_neurons, device=self.device) * (-65.0)
        else:
            v = v_init.to(self.device)

        # Initialize gating variables at steady state
        m = self._alpha_m(v) / (self._alpha_m(v) + self._beta_m(v))
        h = self._alpha_h(v) / (self._alpha_h(v) + self._beta_h(v))
        n = self._alpha_n(v) / (self._alpha_n(v) + self._beta_n(v))

        voltages_list = []
        m_list = []
        h_list = []
        n_list = []

        for t in range(n_steps):
            # Ion currents
            i_na = self.g_na * (m ** 3) * h * (v - self.e_na)
            i_k = self.g_k * (n ** 4) * (v - self.e_k)
            i_l = self.g_l * (v - self.e_l)

            # Membrane equation
            dv = (-i_na - i_k - i_l + input_current[:, t, :]) / self.c_m * dt

            # Gating variables
            dm = (self._alpha_m(v) * (1 - m) - self._beta_m(v) * m) * dt
            dh = (self._alpha_h(v) * (1 - h) - self._beta_h(v) * h) * dt
            dn = (self._alpha_n(v) * (1 - n) - self._beta_n(v) * n) * dt

            # Update
            v = v + dv
            m = m + dm
            h = h + dh
            n = n + dn

            # Clamp gating variables
            m = torch.clamp(m, 0, 1)
            h = torch.clamp(h, 0, 1)
            n = torch.clamp(n, 0, 1)

            voltages_list.append(v)
            m_list.append(m)
            h_list.append(h)
            n_list.append(n)

        voltages = torch.stack(voltages_list, dim=1)

        # Detect spikes (voltage crossings of 0 mV)
        spikes = (voltages[:, 1:, :] >= 0) & (voltages[:, :-1, :] < 0)
        spikes = F.pad(spikes.float(), (0, 0, 1, 0), value=0.0)  # Pad first timestep

        gates = {
            'm': torch.stack(m_list, dim=1),
            'h': torch.stack(h_list, dim=1),
            'n': torch.stack(n_list, dim=1),
        }

        if n_neurons == 1:
            spikes = spikes.squeeze(2)
            voltages = voltages.squeeze(2)
            gates = {k: v.squeeze(2) for k, v in gates.items()}

        return spikes, voltages, gates


class SpikingNeuralNetwork(nn.Module):
    """
    Full spiking neural network with configurable neuron types and connectivity.

    Supports recurrent connections, Dale's law, and multiple neuron models.

    Args:
        n_neurons: Number of neurons
        neuron_type: Type of neuron ('lif', 'izhikevich', 'hh')
        connectivity: Connection matrix [n_neurons, n_neurons] (optional)
        enforce_dales: Whether to enforce excitatory/inhibitory separation
        ei_ratio: Fraction of excitatory neurons (default: 0.8)
        device: Torch device

    Example:
        >>> snn = SpikingNeuralNetwork(
        ...     n_neurons=100,
        ...     neuron_type='lif',
        ...     enforce_dales=True
        ... )
        >>> spikes_out, voltages = snn(input_spikes, n_steps=1000)
    """

    def __init__(
        self,
        n_neurons: int,
        neuron_type: str = 'lif',
        connectivity: Optional[Tensor] = None,
        enforce_dales: bool = True,
        ei_ratio: float = 0.8,
        device: Optional[str] = None,
    ):
        super().__init__()

        self.n_neurons = n_neurons
        self.neuron_type = neuron_type
        self.enforce_dales = enforce_dales
        self.ei_ratio = ei_ratio
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # Create neuron model
        if neuron_type == 'lif':
            self.neuron = LeakyIntegrateFireNeuron(device=self.device)
        elif neuron_type == 'izhikevich':
            self.neuron = IzhikevichNeuron(device=self.device)
        elif neuron_type == 'hh':
            self.neuron = HodgkinHuxleyNeuron(device=self.device)
        else:
            raise ValueError(f"Unknown neuron type: {neuron_type}")

        # Connectivity matrix (synaptic weights)
        if connectivity is None:
            # Random sparse connectivity
            connectivity = torch.randn(n_neurons, n_neurons) * 0.1
            mask = torch.rand(n_neurons, n_neurons) < 0.1  # 10% connectivity
            connectivity = connectivity * mask.float()

        self.connectivity = nn.Parameter(connectivity.to(self.device))

        # Dale's law: E/I separation
        if enforce_dales:
            n_exc = int(n_neurons * ei_ratio)
            self.register_buffer('neuron_types', torch.cat([
                torch.ones(n_exc, dtype=torch.bool),
                torch.zeros(n_neurons - n_exc, dtype=torch.bool)
            ]).to(self.device))
        else:
            self.register_buffer('neuron_types', None)

        self.to(self.device)

    def _apply_dales_law(self):
        """Enforce Dale's law on connectivity."""
        if self.enforce_dales and self.neuron_types is not None:
            with torch.no_grad():
                # E neurons: all outgoing weights >= 0
                # I neurons: all outgoing weights <= 0
                for i in range(self.n_neurons):
                    if self.neuron_types[i]:  # Excitatory
                        self.connectivity.data[i, :] = self.connectivity.data[i, :].abs()
                    else:  # Inhibitory
                        self.connectivity.data[i, :] = -self.connectivity.data[i, :].abs()

    def forward(
        self,
        external_input: Tensor,
        n_steps: int,
        dt: float = 1.0,
    ) -> Tuple[Tensor, Tensor]:
        """
        Simulate spiking network.

        Args:
            external_input: External input current [batch, n_steps, n_neurons]
            n_steps: Number of simulation steps
            dt: Time step size

        Returns:
            spikes: Output spikes [batch, n_steps, n_neurons]
            voltages: Membrane potentials [batch, n_steps, n_neurons]
        """
        batch_size = external_input.size(0)

        # Apply Dale's law
        if self.enforce_dales:
            self._apply_dales_law()

        # Storage
        all_spikes = []
        all_voltages = []

        # Initial state
        v = None

        # Recurrent simulation
        for t in range(n_steps):
            # Combine external input with recurrent input
            if t == 0:
                total_input = external_input[:, t, :]
            else:
                # Recurrent connections: I_rec = W @ spikes_{t-1}
                prev_spikes = all_spikes[-1]
                recurrent_input = prev_spikes @ self.connectivity.T
                total_input = external_input[:, t, :] + recurrent_input

            # Simulate one step
            if self.neuron_type in ['lif', 'izhikevich']:
                spikes, voltages = self.neuron(
                    total_input.unsqueeze(1),  # [batch, 1, n_neurons]
                    dt=dt,
                    v_init=v
                )
                spikes = spikes[:, 0, :]  # [batch, n_neurons]
                voltages = voltages[:, 0, :]
                v = voltages  # Update state
            else:  # HH
                spikes, voltages, _ = self.neuron(
                    total_input.unsqueeze(1),
                    dt=dt,
                    v_init=v
                )
                spikes = spikes[:, 0, :]
                voltages = voltages[:, 0, :]
                v = voltages

            all_spikes.append(spikes)
            all_voltages.append(voltages)

        # Stack
        spikes_out = torch.stack(all_spikes, dim=1)
        voltages_out = torch.stack(all_voltages, dim=1)

        return spikes_out, voltages_out
