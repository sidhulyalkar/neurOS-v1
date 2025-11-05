"""
Optogenetic Interventions

Models optogenetic tools for precise temporal and spatial control of neural activity.
Includes channelrhodopsins, halorhodopsins, and archaerhodopsins.

References:
- Boyden et al. (2005): Millisecond-timescale optical control
- Zhang et al. (2007): Multimodal fast optical interrogation
- Deisseroth (2015): Optogenetics: 10 years
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Callable, Dict
from dataclasses import dataclass
import numpy as np


@dataclass
class OptogeneticParameters:
    """Parameters for optogenetic stimulation."""
    wavelength: float  # nm
    power_density: float  # mW/mm^2
    pulse_width: float  # ms
    frequency: float  # Hz
    duty_cycle: float = 0.5  # Fraction of time light is on


class Opsin(nn.Module):
    """Base class for light-sensitive opsins."""

    def __init__(
        self,
        g_max: float,  # Maximum conductance (nS)
        E_rev: float,  # Reversal potential (mV)
        tau_on: float,  # Activation time constant (ms)
        tau_off: float,  # Deactivation time constant (ms)
        wavelength_peak: float,  # Peak sensitivity (nm)
        dt: float = 0.1
    ):
        super().__init__()
        self.g_max = g_max
        self.E_rev = E_rev
        self.tau_on = tau_on
        self.tau_off = tau_off
        self.wavelength_peak = wavelength_peak
        self.dt = dt

        # Opsin state (fraction activated)
        self.register_buffer('O', None)

    def light_sensitivity(self, wavelength: torch.Tensor) -> torch.Tensor:
        """
        Wavelength-dependent activation.

        Gaussian sensitivity around peak wavelength.

        Args:
            wavelength: Light wavelength (nm)

        Returns:
            Sensitivity [0, 1]
        """
        sigma = 50.0  # nm (half-width)
        return torch.exp(-((wavelength - self.wavelength_peak) ** 2) / (2 * sigma ** 2))

    def photocurrent(
        self,
        V: torch.Tensor,
        light_intensity: torch.Tensor,
        wavelength: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute photocurrent.

        Args:
            V: Membrane potential (mV)
            light_intensity: Light power density (mW/mm^2)
            wavelength: Wavelength (nm)

        Returns:
            I_photo: Photocurrent (pA)
        """
        if self.O is None:
            self.O = torch.zeros_like(V)

        # Light-dependent activation
        sensitivity = self.light_sensitivity(wavelength)
        activation = sensitivity * torch.tanh(light_intensity / 10.0)

        # State dynamics
        dO = (activation - self.O) / torch.where(
            light_intensity > 0,
            torch.tensor(self.tau_on, device=V.device),
            torch.tensor(self.tau_off, device=V.device)
        )

        self.O = self.O + self.dt * dO
        self.O = torch.clamp(self.O, 0.0, 1.0)

        # Current
        g = self.g_max * self.O
        I = g * (V - self.E_rev)  # pA

        return I

    def reset(self):
        """Reset opsin state."""
        if self.O is not None:
            self.O.zero_()


class ChR2(Opsin):
    """
    Channelrhodopsin-2 (ChR2).

    Excitatory optogenetic actuator. Blue light activates cation channel.

    References:
        Nagel et al. (2003), PNAS
    """

    def __init__(
        self,
        g_max: float = 1000.0,  # nS
        E_rev: float = 0.0,     # mV (non-selective cation)
        tau_on: float = 1.0,    # ms
        tau_off: float = 10.0,  # ms
        wavelength_peak: float = 470.0,  # nm (blue)
        dt: float = 0.1
    ):
        super().__init__(g_max, E_rev, tau_on, tau_off, wavelength_peak, dt)


class ChR2_H134R(Opsin):
    """
    ChR2-H134R variant.

    Enhanced photocurrents, slower off kinetics.
    """

    def __init__(
        self,
        g_max: float = 2000.0,
        E_rev: float = 0.0,
        tau_on: float = 1.0,
        tau_off: float = 20.0,  # Slower deactivation
        wavelength_peak: float = 470.0,
        dt: float = 0.1
    ):
        super().__init__(g_max, E_rev, tau_on, tau_off, wavelength_peak, dt)


class ChETA(Opsin):
    """
    Channelrhodopsin with Enhanced Temporal Accuracy (ChETA).

    Fast on/off kinetics for high-frequency stimulation.

    References:
        Gunaydin et al. (2010), Nat. Neurosci.
    """

    def __init__(
        self,
        g_max: float = 800.0,
        E_rev: float = 0.0,
        tau_on: float = 0.5,   # Fast
        tau_off: float = 3.0,  # Fast
        wavelength_peak: float = 470.0,
        dt: float = 0.1
    ):
        super().__init__(g_max, E_rev, tau_on, tau_off, wavelength_peak, dt)


class ReaChR(Opsin):
    """
    Red-Activatable Channelrhodopsin (ReaChR).

    Red light activation for deeper tissue penetration.

    References:
        Lin et al. (2013), Nat. Neurosci.
    """

    def __init__(
        self,
        g_max: float = 600.0,
        E_rev: float = 0.0,
        tau_on: float = 2.0,
        tau_off: float = 15.0,
        wavelength_peak: float = 590.0,  # nm (red)
        dt: float = 0.1
    ):
        super().__init__(g_max, E_rev, tau_on, tau_off, wavelength_peak, dt)


class NpHR(Opsin):
    """
    Halorhodopsin from Natronomonas pharaonis (NpHR).

    Inhibitory optogenetic actuator. Yellow light activates Cl- pump.

    References:
        Zhang et al. (2007), Nature
    """

    def __init__(
        self,
        g_max: float = 500.0,
        E_rev: float = -65.0,  # mV (chloride reversal)
        tau_on: float = 1.0,
        tau_off: float = 15.0,
        wavelength_peak: float = 580.0,  # nm (yellow)
        dt: float = 0.1
    ):
        super().__init__(g_max, E_rev, tau_on, tau_off, wavelength_peak, dt)


class ArchT(Opsin):
    """
    Archaerhodopsin-3 from Halorubrum sodomense (Arch/ArchT).

    Inhibitory outward proton pump. Green light activation.

    References:
        Chow et al. (2010), Nature
    """

    def __init__(
        self,
        g_max: float = 400.0,
        E_rev: float = -100.0,  # Hyperpolarizing
        tau_on: float = 0.5,    # Fast
        tau_off: float = 10.0,
        wavelength_peak: float = 570.0,  # nm (green-yellow)
        dt: float = 0.1
    ):
        super().__init__(g_max, E_rev, tau_on, tau_off, wavelength_peak, dt)


class eNpHR3(Opsin):
    """
    Enhanced NpHR 3.0.

    Improved membrane trafficking and photocurrents.
    """

    def __init__(
        self,
        g_max: float = 800.0,  # Enhanced
        E_rev: float = -65.0,
        tau_on: float = 1.0,
        tau_off: float = 12.0,
        wavelength_peak: float = 590.0,
        dt: float = 0.1
    ):
        super().__init__(g_max, E_rev, tau_on, tau_off, wavelength_peak, dt)


class OptoStimulator(nn.Module):
    """
    Optogenetic stimulation controller.

    Generates light pulses with specified temporal patterns.
    """

    def __init__(
        self,
        opsin: Opsin,
        params: OptogeneticParameters,
        spatial_profile: Optional[Callable] = None
    ):
        super().__init__()
        self.opsin = opsin
        self.params = params
        self.spatial_profile = spatial_profile

        # Stimulation state
        self.register_buffer('time', None)
        self.register_buffer('pulse_phase', None)

    def generate_pulse_train(
        self,
        duration: float,
        dt: float
    ) -> torch.Tensor:
        """
        Generate pulsed light stimulation.

        Args:
            duration: Total duration (ms)
            dt: Time step (ms)

        Returns:
            Light intensity time series (n_steps,)
        """
        n_steps = int(duration / dt)
        time = torch.arange(n_steps, dtype=torch.float32) * dt

        # Pulse train
        period = 1000.0 / self.params.frequency  # ms
        pulse_on = time % period < self.params.pulse_width

        intensity = torch.where(
            pulse_on,
            torch.tensor(self.params.power_density),
            torch.zeros_like(time)
        )

        return intensity

    def spatial_illumination(
        self,
        positions: torch.Tensor,
        light_center: torch.Tensor,
        beam_width: float
    ) -> torch.Tensor:
        """
        Compute spatial light distribution (Gaussian beam).

        Args:
            positions: Neuron positions (n_neurons, 3)
            light_center: Center of light beam (3,)
            beam_width: Gaussian width (μm)

        Returns:
            Relative intensity at each position (n_neurons,)
        """
        distances = torch.norm(positions - light_center, dim=1)
        intensity = torch.exp(-(distances ** 2) / (2 * beam_width ** 2))

        return intensity

    def forward(
        self,
        V: torch.Tensor,
        light_on: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        light_center: Optional[torch.Tensor] = None,
        beam_width: float = 100.0
    ) -> torch.Tensor:
        """
        Apply optogenetic stimulation.

        Args:
            V: Membrane potential (batch, n_neurons)
            light_on: Binary indicator of light (batch,) or (batch, n_neurons)
            positions: Neuron positions for spatial selectivity
            light_center: Center of light beam
            beam_width: Beam width (μm)

        Returns:
            Photocurrent (batch, n_neurons)
        """
        batch_size, n_neurons = V.shape

        # Spatial modulation
        if positions is not None and light_center is not None:
            spatial_mod = self.spatial_illumination(positions, light_center, beam_width)
            spatial_mod = spatial_mod.unsqueeze(0).expand(batch_size, -1)
        else:
            spatial_mod = torch.ones_like(V)

        # Light intensity
        if light_on.dim() == 1:
            light_on = light_on.unsqueeze(1).expand(-1, n_neurons)

        intensity = light_on.float() * self.params.power_density * spatial_mod

        # Wavelength
        wavelength = torch.full_like(intensity, self.params.wavelength)

        # Photocurrent
        I_photo = self.opsin.photocurrent(V, intensity, wavelength)

        return I_photo

    def reset(self):
        """Reset stimulator state."""
        self.opsin.reset()


class OptogeneticExperiment:
    """
    Design and analyze optogenetic experiments.

    Supports common paradigms:
    - Activation/inhibition
    - Frequency tuning
    - Spatiotemporal patterns
    - Photostimulation mapping
    """

    def __init__(self, dt: float = 0.1):
        self.dt = dt

    def create_pulse_train(
        self,
        frequency: float,
        pulse_width: float,
        duration: float,
        power: float,
        wavelength: float
    ) -> OptogeneticParameters:
        """
        Create parameters for pulse train stimulation.

        Args:
            frequency: Stimulation frequency (Hz)
            pulse_width: Pulse duration (ms)
            duration: Total duration (ms)
            power: Light power density (mW/mm^2)
            wavelength: Light wavelength (nm)

        Returns:
            Optogenetic parameters
        """
        duty_cycle = (pulse_width * frequency) / 1000.0

        return OptogeneticParameters(
            wavelength=wavelength,
            power_density=power,
            pulse_width=pulse_width,
            frequency=frequency,
            duty_cycle=duty_cycle
        )

    def ramp_stimulation(
        self,
        start_power: float,
        end_power: float,
        duration: float,
        wavelength: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create ramping light intensity.

        Args:
            start_power: Initial power (mW/mm^2)
            end_power: Final power (mW/mm^2)
            duration: Duration (ms)
            wavelength: Wavelength (nm)

        Returns:
            (time, intensity): Time points and intensity values
        """
        n_steps = int(duration / self.dt)
        time = torch.arange(n_steps, dtype=torch.float32) * self.dt
        intensity = torch.linspace(start_power, end_power, n_steps)

        return time, intensity

    def photostimulation_map(
        self,
        n_locations: int,
        stim_duration: float,
        interval: float,
        power: float,
        wavelength: float
    ) -> Dict[str, torch.Tensor]:
        """
        Create photostimulation mapping protocol.

        Systematically stimulate different locations to map connectivity.

        Args:
            n_locations: Number of stimulation sites
            stim_duration: Duration per location (ms)
            interval: Inter-stimulus interval (ms)
            power: Light power (mW/mm^2)
            wavelength: Wavelength (nm)

        Returns:
            Stimulation protocol
        """
        total_duration = n_locations * (stim_duration + interval)
        n_steps = int(total_duration / self.dt)

        time = torch.arange(n_steps, dtype=torch.float32) * self.dt
        location_id = torch.zeros(n_steps, dtype=torch.long)
        intensity = torch.zeros(n_steps)

        for loc in range(n_locations):
            start_time = loc * (stim_duration + interval)
            end_time = start_time + stim_duration

            start_idx = int(start_time / self.dt)
            end_idx = int(end_time / self.dt)

            location_id[start_idx:end_idx] = loc
            intensity[start_idx:end_idx] = power

        return {
            'time': time,
            'location': location_id,
            'intensity': intensity,
            'wavelength': wavelength
        }

    def measure_response_latency(
        self,
        spike_times: torch.Tensor,
        stim_time: float,
        window: float = 100.0
    ) -> Tuple[float, float]:
        """
        Measure response latency to photostimulation.

        Args:
            spike_times: Spike times (ms)
            stim_time: Stimulation onset (ms)
            window: Search window after stim (ms)

        Returns:
            (latency, jitter): Mean and std of first spike latency
        """
        # Find first spike after stimulus
        post_stim_spikes = spike_times[
            (spike_times > stim_time) & (spike_times < stim_time + window)
        ]

        if len(post_stim_spikes) == 0:
            return float('nan'), float('nan')

        latencies = post_stim_spikes - stim_time

        return latencies.min().item(), latencies.std().item()

    def frequency_tuning_curve(
        self,
        frequencies: list,
        neuron_response_fn: Callable
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Measure frequency response to optogenetic stimulation.

        Args:
            frequencies: List of stimulation frequencies (Hz)
            neuron_response_fn: Function that returns firing rate given frequency

        Returns:
            (frequencies, responses): Tuning curve
        """
        freqs = torch.tensor(frequencies, dtype=torch.float32)
        responses = torch.zeros_like(freqs)

        for i, freq in enumerate(freqs):
            responses[i] = neuron_response_fn(freq)

        return freqs, responses


def select_opsin(
    goal: str,
    speed: str = 'medium',
    wavelength: Optional[str] = None
) -> Opsin:
    """
    Select appropriate opsin for experimental goal.

    Args:
        goal: 'excitation' or 'inhibition'
        speed: 'fast', 'medium', or 'slow'
        wavelength: 'blue', 'green', 'yellow', 'red', or None (auto)

    Returns:
        Opsin instance
    """
    if goal == 'excitation':
        if speed == 'fast':
            return ChETA()
        elif wavelength == 'red':
            return ReaChR()
        elif wavelength == 'blue' or wavelength is None:
            if speed == 'medium':
                return ChR2()
            else:
                return ChR2_H134R()
        else:
            return ChR2()

    elif goal == 'inhibition':
        if wavelength == 'green':
            return ArchT()
        elif wavelength == 'yellow' or wavelength is None:
            if speed == 'fast':
                return NpHR()
            else:
                return eNpHR3()
        else:
            return NpHR()

    else:
        raise ValueError(f"Unknown goal: {goal}. Use 'excitation' or 'inhibition'.")
