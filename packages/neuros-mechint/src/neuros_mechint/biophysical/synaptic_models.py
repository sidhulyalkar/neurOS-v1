"""
Synaptic Plasticity Models

Biophysical models of synaptic learning rules including STDP, homeostatic
plasticity, and metaplasticity.

References:
- Bi & Poo (1998): Synaptic modifications by correlated activity
- Abbott & Nelson (2000): Synaptic plasticity: taming the beast
- Zenke et al. (2013): Synaptic plasticity in neural networks needs homeostasis
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Callable, Dict
from dataclasses import dataclass


@dataclass
class STDPParameters:
    """Parameters for STDP learning rule."""
    A_plus: float = 0.01   # LTP amplitude
    A_minus: float = 0.0105  # LTD amplitude
    tau_plus: float = 20.0  # LTP time constant (ms)
    tau_minus: float = 20.0  # LTD time constant (ms)
    w_min: float = 0.0     # Minimum weight
    w_max: float = 1.0     # Maximum weight


class STDP(nn.Module):
    """
    Spike-Timing-Dependent Plasticity (STDP).

    Hebbian learning rule where relative timing determines
    potentiation (LTP) or depression (LTD).

    Δw = A_plus * exp(-Δt / tau_plus)   if Δt > 0 (pre before post)
    Δw = -A_minus * exp(Δt / tau_minus)  if Δt < 0 (post before pre)

    References:
        Bi & Poo (1998), J. Neurosci.
    """

    def __init__(
        self,
        params: Optional[STDPParameters] = None,
        dt: float = 1.0  # Time step (ms)
    ):
        super().__init__()
        if params is None:
            params = STDPParameters()

        self.params = params
        self.dt = dt

        # Eligibility traces
        self.register_buffer('pre_trace', None)
        self.register_buffer('post_trace', None)

    def forward(
        self,
        weights: torch.Tensor,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor
    ) -> torch.Tensor:
        """
        Update synaptic weights based on pre/post spike times.

        Args:
            weights: Current weights (n_pre, n_post)
            pre_spikes: Presynaptic spikes (batch, n_pre)
            post_spikes: Postsynaptic spikes (batch, n_post)

        Returns:
            Updated weights
        """
        batch_size, n_pre = pre_spikes.shape
        n_post = post_spikes.shape[1]
        device = weights.device

        # Initialize traces
        if self.pre_trace is None or self.pre_trace.shape != (batch_size, n_pre):
            self.pre_trace = torch.zeros(batch_size, n_pre, device=device)
            self.post_trace = torch.zeros(batch_size, n_post, device=device)

        # Update traces
        decay_pre = torch.exp(torch.tensor(-self.dt / self.params.tau_plus, device=device))
        decay_post = torch.exp(torch.tensor(-self.dt / self.params.tau_minus, device=device))

        self.pre_trace = self.pre_trace * decay_pre + pre_spikes
        self.post_trace = self.post_trace * decay_post + post_spikes

        # Weight updates
        # LTP: post spike looks back at pre trace
        dw_ltp = self.params.A_plus * torch.einsum('bi,bj->ij', self.pre_trace, post_spikes)

        # LTD: pre spike looks back at post trace
        dw_ltd = -self.params.A_minus * torch.einsum('bi,bj->ij', pre_spikes, self.post_trace)

        # Total update
        dw = (dw_ltp + dw_ltd) / batch_size

        # Update weights with bounds
        new_weights = weights + dw
        new_weights = torch.clamp(new_weights, self.params.w_min, self.params.w_max)

        return new_weights

    def reset(self):
        """Reset traces."""
        if self.pre_trace is not None:
            self.pre_trace.zero_()
            self.post_trace.zero_()


class TripletSTDP(nn.Module):
    """
    Triplet STDP rule.

    Extends pairwise STDP to account for triplet interactions,
    better explaining experimental data.

    References:
        Pfister & Gerstner (2006), J. Neurosci.
    """

    def __init__(
        self,
        A2_plus: float = 0.01,
        A2_minus: float = 0.0105,
        A3_plus: float = 0.01,
        A3_minus: float = 0.0105,
        tau_plus: float = 20.0,
        tau_minus: float = 20.0,
        tau_x: float = 100.0,
        tau_y: float = 100.0,
        dt: float = 1.0
    ):
        super().__init__()
        self.A2_plus = A2_plus
        self.A2_minus = A2_minus
        self.A3_plus = A3_plus
        self.A3_minus = A3_minus
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.tau_x = tau_x
        self.tau_y = tau_y
        self.dt = dt

        # Traces
        self.register_buffer('r1', None)  # Fast pre trace
        self.register_buffer('r2', None)  # Slow pre trace
        self.register_buffer('o1', None)  # Fast post trace
        self.register_buffer('o2', None)  # Slow post trace

    def forward(
        self,
        weights: torch.Tensor,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor
    ) -> torch.Tensor:
        """
        Update weights with triplet STDP.

        Args:
            weights: Synaptic weights (n_pre, n_post)
            pre_spikes: Presynaptic spikes (batch, n_pre)
            post_spikes: Postsynaptic spikes (batch, n_post)

        Returns:
            Updated weights
        """
        batch_size, n_pre = pre_spikes.shape
        n_post = post_spikes.shape[1]
        device = weights.device

        # Initialize traces
        if self.r1 is None or self.r1.shape != (batch_size, n_pre):
            self.r1 = torch.zeros(batch_size, n_pre, device=device)
            self.r2 = torch.zeros(batch_size, n_pre, device=device)
            self.o1 = torch.zeros(batch_size, n_post, device=device)
            self.o2 = torch.zeros(batch_size, n_post, device=device)

        # Decay traces
        self.r1 = self.r1 * torch.exp(torch.tensor(-self.dt / self.tau_plus))
        self.r2 = self.r2 * torch.exp(torch.tensor(-self.dt / self.tau_x))
        self.o1 = self.o1 * torch.exp(torch.tensor(-self.dt / self.tau_minus))
        self.o2 = self.o2 * torch.exp(torch.tensor(-self.dt / self.tau_y))

        # LTP (pairwise + triplet)
        dw_ltp = torch.einsum(
            'bi,bj->ij',
            self.r1,
            post_spikes * (self.A2_plus + self.A3_plus * self.o2)
        )

        # LTD (pairwise + triplet)
        dw_ltd = torch.einsum(
            'bi,bj->ij',
            pre_spikes * self.r2,
            self.o1 * (self.A2_minus + self.A3_minus * self.r2)
        )

        # Update traces
        self.r1 = self.r1 + pre_spikes
        self.r2 = self.r2 + pre_spikes
        self.o1 = self.o1 + post_spikes
        self.o2 = self.o2 + post_spikes

        # Update weights
        dw = (dw_ltp - dw_ltd) / batch_size
        new_weights = torch.clamp(weights + dw, 0.0, 1.0)

        return new_weights

    def reset(self):
        """Reset all traces."""
        if self.r1 is not None:
            self.r1.zero_()
            self.r2.zero_()
            self.o1.zero_()
            self.o2.zero_()


class ShortTermPlasticity(nn.Module):
    """
    Short-term synaptic plasticity (facilitation and depression).

    Models vesicle depletion and calcium-dependent release probability.

    References:
        Tsodyks & Markram (1997), PNAS
    """

    def __init__(
        self,
        U: float = 0.5,      # Baseline release probability
        tau_rec: float = 800.0,  # Recovery time constant (ms)
        tau_facil: float = 0.0,  # Facilitation time constant (ms)
        dt: float = 1.0
    ):
        super().__init__()
        self.U = U
        self.tau_rec = tau_rec
        self.tau_facil = tau_facil if tau_facil > 0 else 1e-6  # Avoid division by zero
        self.dt = dt

        # Dynamic variables
        self.register_buffer('x', None)  # Fraction of available resources
        self.register_buffer('u', None)  # Release probability

    def forward(
        self,
        pre_spikes: torch.Tensor,
        weights: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute effective synaptic strength with STP.

        Args:
            pre_spikes: Presynaptic spikes (batch, n_pre)
            weights: Static weights (n_pre, n_post)

        Returns:
            (effective_weights, psr): Effective weights and postsynaptic response
        """
        batch_size, n_pre = pre_spikes.shape
        device = weights.device

        # Initialize
        if self.x is None or self.x.shape != (batch_size, n_pre):
            self.x = torch.ones(batch_size, n_pre, device=device)
            self.u = torch.full((batch_size, n_pre), self.U, device=device)

        # Decay dynamics
        dx = (1.0 - self.x) / self.tau_rec
        du = (self.U - self.u) / self.tau_facil

        # Update on spike
        u_before = self.u.clone()
        x_before = self.x.clone()

        # Facilitation
        self.u = self.u + self.dt * du
        self.u = torch.where(pre_spikes > 0, self.u + self.U * (1.0 - self.u), self.u)

        # Depression
        self.x = self.x + self.dt * dx
        released = u_before * x_before * pre_spikes
        self.x = torch.where(pre_spikes > 0, self.x - released, self.x)

        # Effective weights
        effective_weights = weights.unsqueeze(0) * released.unsqueeze(-1)

        # Postsynaptic response
        psr = torch.einsum('bi,bij->bj', pre_spikes, effective_weights)

        return effective_weights.mean(0), psr

    def reset(self):
        """Reset to baseline."""
        if self.x is not None:
            self.x.fill_(1.0)
            self.u.fill_(self.U)


class SynapticDynamics(nn.Module):
    """
    Combined long-term and short-term synaptic plasticity.

    Integrates STDP for learning with STP for dynamics.
    """

    def __init__(
        self,
        stdp_params: Optional[STDPParameters] = None,
        U: float = 0.5,
        tau_rec: float = 800.0,
        tau_facil: float = 0.0,
        dt: float = 1.0
    ):
        super().__init__()
        self.stdp = STDP(stdp_params, dt=dt)
        self.stp = ShortTermPlasticity(U, tau_rec, tau_facil, dt=dt)

    def forward(
        self,
        weights: torch.Tensor,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        learn: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update weights and compute synaptic output.

        Args:
            weights: Long-term weights (n_pre, n_post)
            pre_spikes: Presynaptic spikes (batch, n_pre)
            post_spikes: Postsynaptic spikes (batch, n_post)
            learn: Whether to apply STDP

        Returns:
            (new_weights, psr): Updated weights and postsynaptic response
        """
        # Short-term dynamics
        _, psr = self.stp.forward(pre_spikes, weights)

        # Long-term plasticity
        if learn:
            new_weights = self.stdp.forward(weights, pre_spikes, post_spikes)
        else:
            new_weights = weights

        return new_weights, psr

    def reset(self):
        """Reset both STDP and STP."""
        self.stdp.reset()
        self.stp.reset()


class HomeostaticPlasticity(nn.Module):
    """
    Homeostatic synaptic scaling.

    Maintains neuronal firing rate around a target value by
    multiplicatively scaling all synaptic weights.

    References:
        Turrigiano & Nelson (2004), Nat. Rev. Neurosci.
    """

    def __init__(
        self,
        target_rate: float = 10.0,  # Hz
        tau_homeostatic: float = 10000.0,  # ms (slow)
        dt: float = 1.0
    ):
        super().__init__()
        self.target_rate = target_rate / 1000.0  # Convert to spikes/ms
        self.tau_homeostatic = tau_homeostatic
        self.dt = dt

        # Running average of firing rate
        self.register_buffer('avg_rate', None)

    def forward(
        self,
        weights: torch.Tensor,
        post_spikes: torch.Tensor
    ) -> torch.Tensor:
        """
        Scale weights to maintain target firing rate.

        Args:
            weights: Synaptic weights (n_pre, n_post)
            post_spikes: Postsynaptic spikes (batch, n_post)

        Returns:
            Scaled weights
        """
        batch_size, n_post = post_spikes.shape
        device = weights.device

        # Initialize running average
        if self.avg_rate is None or self.avg_rate.shape[0] != n_post:
            self.avg_rate = torch.full((n_post,), self.target_rate, device=device)

        # Update running average
        instantaneous_rate = post_spikes.mean(0) / self.dt  # spikes/ms
        alpha = self.dt / self.tau_homeostatic
        self.avg_rate = (1 - alpha) * self.avg_rate + alpha * instantaneous_rate

        # Compute scaling factor
        scale = (self.target_rate / (self.avg_rate + 1e-6))

        # Scale weights for each postsynaptic neuron
        scaled_weights = weights * scale.unsqueeze(0)

        return scaled_weights

    def reset(self):
        """Reset running average."""
        if self.avg_rate is not None:
            self.avg_rate.fill_(self.target_rate)


class BCMRule(nn.Module):
    """
    Bienenstock-Cooper-Munro (BCM) learning rule.

    Rate-based plasticity rule with sliding threshold for stability.

    dw/dt = η * post * (post - θ) * pre

    References:
        Bienenstock et al. (1982), J. Neurosci.
    """

    def __init__(
        self,
        learning_rate: float = 1e-4,
        tau_theta: float = 10000.0,  # Threshold time constant (ms)
        dt: float = 1.0
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.tau_theta = tau_theta
        self.dt = dt

        # Sliding threshold
        self.register_buffer('theta', None)
        self.register_buffer('avg_post', None)

    def forward(
        self,
        weights: torch.Tensor,
        pre_rate: torch.Tensor,
        post_rate: torch.Tensor
    ) -> torch.Tensor:
        """
        Update weights with BCM rule.

        Args:
            weights: Synaptic weights (n_pre, n_post)
            pre_rate: Presynaptic firing rates (batch, n_pre)
            post_rate: Postsynaptic firing rates (batch, n_post)

        Returns:
            Updated weights
        """
        batch_size, n_post = post_rate.shape
        device = weights.device

        # Initialize threshold
        if self.theta is None or self.theta.shape[0] != n_post:
            self.theta = torch.ones(n_post, device=device)
            self.avg_post = torch.zeros(n_post, device=device)

        # Update threshold: θ = E[post^2]
        alpha = self.dt / self.tau_theta
        self.avg_post = (1 - alpha) * self.avg_post + alpha * post_rate.mean(0)
        self.theta = self.avg_post ** 2

        # BCM update
        post_centered = post_rate - self.theta.unsqueeze(0)
        dw = self.learning_rate * torch.einsum('bi,bj->ij', pre_rate, post_rate * post_centered)
        dw = dw / batch_size

        new_weights = torch.clamp(weights + dw, 0.0, None)

        return new_weights

    def reset(self):
        """Reset threshold."""
        if self.theta is not None:
            self.theta.fill_(1.0)
            self.avg_post.zero_()


class Metaplasticity(nn.Module):
    """
    Metaplasticity: plasticity of plasticity.

    Learning rate modulated by prior activity history.

    References:
        Abraham & Bear (1996), Trends Neurosci.
    """

    def __init__(
        self,
        base_lr: float = 1e-3,
        tau_meta: float = 100000.0,  # Very slow (ms)
        dt: float = 1.0
    ):
        super().__init__()
        self.base_lr = base_lr
        self.tau_meta = tau_meta
        self.dt = dt

        # Metaplastic state
        self.register_buffer('meta_state', None)

    def forward(
        self,
        weights: torch.Tensor,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        plasticity_rule: Callable
    ) -> torch.Tensor:
        """
        Modulate plasticity based on activity history.

        Args:
            weights: Synaptic weights
            pre_spikes: Presynaptic spikes
            post_spikes: Postsynaptic spikes
            plasticity_rule: Base plasticity rule (e.g., STDP)

        Returns:
            Updated weights
        """
        n_pre, n_post = weights.shape
        device = weights.device

        # Initialize metaplastic state
        if self.meta_state is None or self.meta_state.shape != (n_pre, n_post):
            self.meta_state = torch.ones(n_pre, n_post, device=device)

        # Update metaplastic state based on activity
        activity = torch.einsum('bi,bj->ij', pre_spikes, post_spikes)
        alpha = self.dt / self.tau_meta
        self.meta_state = (1 - alpha) * self.meta_state + alpha * (1.0 + activity)

        # Modulate learning rate
        effective_lr = self.base_lr / self.meta_state

        # Apply base plasticity rule with modulated learning rate
        # (This is a simplified version - in practice would need to pass effective_lr to rule)
        new_weights = plasticity_rule(weights, pre_spikes, post_spikes)

        return new_weights

    def reset(self):
        """Reset metaplastic state."""
        if self.meta_state is not None:
            self.meta_state.fill_(1.0)
