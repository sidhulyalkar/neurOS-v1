"""
Neural Stimulation Methods

Transcranial magnetic stimulation (TMS), deep brain stimulation (DBS),
and electrical stimulation for neural circuit manipulation.

References:
- Hallett (2007): Transcranial magnetic stimulation
- Benabid (2003): Deep brain stimulation
- Fritsch & Hitzig (1870): Electrical brain stimulation
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum
import numpy as np


class StimulationType(Enum):
    """Type of stimulation."""
    TMS = "transcranial_magnetic"
    DBS = "deep_brain"
    ELECTRICAL = "direct_electrical"
    TDCS = "transcranial_direct_current"
    TACS = "transcranial_alternating_current"


@dataclass
class StimulationParameters:
    """Parameters for neural stimulation."""
    amplitude: float  # Stimulation strength (mA for DBS, V/m for TMS)
    frequency: float  # Hz
    pulse_width: float  # ms
    n_pulses: int  # Number of pulses
    interpulse_interval: float  # ms
    waveform: str = 'biphasic'  # 'monophasic', 'biphasic', 'sinusoidal'


class TMS(nn.Module):
    """
    Transcranial Magnetic Stimulation (TMS).

    Non-invasive magnetic field induction of neural currents.

    Types:
    - Single pulse
    - Paired pulse
    - Repetitive TMS (rTMS)
    - Theta burst stimulation (TBS)
    """

    def __init__(
        self,
        coil_position: torch.Tensor,  # (3,) position
        coil_orientation: torch.Tensor,  # (3,) unit vector
        max_field: float = 1.5,  # Tesla
        dt: float = 0.1
    ):
        super().__init__()
        self.register_buffer('coil_position', coil_position)
        self.register_buffer('coil_orientation', coil_orientation)
        self.max_field = max_field
        self.dt = dt

    def magnetic_field(
        self,
        positions: torch.Tensor,
        amplitude: float,
        time: float
    ) -> torch.Tensor:
        """
        Compute induced magnetic field at neuron positions.

        B(r) ∝ 1/r^2 for magnetic coil

        Args:
            positions: Neuron positions (n_neurons, 3)
            amplitude: Field strength (0-1)
            time: Current time (ms)

        Returns:
            Field strength at each position (n_neurons,)
        """
        # Distance from coil
        displacement = positions - self.coil_position
        distance = torch.norm(displacement, dim=1)

        # Dipole field (simplified)
        # Actual TMS coils have complex field geometry
        field_magnitude = amplitude * self.max_field / (distance ** 2 + 0.1)

        # Directional component (dot product with coil orientation)
        direction_factor = torch.sum(
            displacement / distance.unsqueeze(1) * self.coil_orientation,
            dim=1
        ).abs()

        field = field_magnitude * direction_factor

        return field

    def induced_electric_field(
        self,
        magnetic_field: torch.Tensor,
        dB_dt: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute induced electric field from Faraday's law.

        ∇ × E = -dB/dt

        Args:
            magnetic_field: Magnetic field (Tesla)
            dB_dt: Time derivative of field (T/s)

        Returns:
            Induced electric field (V/m)
        """
        # Simplified: E ∝ -dB/dt
        # Actual computation requires spatial gradients
        E_field = -dB_dt * 1e3  # Convert to V/m

        return E_field

    def forward(
        self,
        positions: torch.Tensor,
        params: StimulationParameters,
        duration: float
    ) -> torch.Tensor:
        """
        Simulate TMS pulse.

        Args:
            positions: Neuron positions (n_neurons, 3)
            params: Stimulation parameters
            duration: Duration (ms)

        Returns:
            Induced current over time (n_steps, n_neurons)
        """
        n_steps = int(duration / self.dt)
        n_neurons = positions.shape[0]

        time = torch.arange(n_steps, dtype=torch.float32) * self.dt
        induced_current = torch.zeros(n_steps, n_neurons)

        # Generate pulse train
        pulse_times = torch.arange(params.n_pulses, dtype=torch.float32) * params.interpulse_interval

        for t_idx, t in enumerate(time):
            # Check if pulse is active
            pulse_active = False
            for pulse_time in pulse_times:
                if pulse_time <= t < pulse_time + params.pulse_width:
                    pulse_active = True
                    break

            if pulse_active:
                # Magnetic field during pulse
                B = self.magnetic_field(positions, params.amplitude, t)

                # Time derivative (simplified)
                dB_dt = B / (params.pulse_width * 1e-3)  # T/s

                # Induced E-field
                E_field = self.induced_electric_field(B, dB_dt)

                # Induced current (I = σ * E, where σ is conductivity)
                conductivity = 0.3  # S/m (brain tissue)
                induced_current[t_idx] = conductivity * E_field

        return induced_current


class DBS(nn.Module):
    """
    Deep Brain Stimulation (DBS).

    Chronic electrical stimulation of deep brain structures.

    Common targets:
    - STN (subthalamic nucleus) for Parkinson's
    - GPi (globus pallidus internus) for dystonia
    - Vim (ventral intermediate nucleus) for tremor
    """

    def __init__(
        self,
        electrode_position: torch.Tensor,  # (3,) position
        electrode_type: str = 'monopolar',
        dt: float = 0.1
    ):
        super().__init__()
        self.register_buffer('electrode_position', electrode_position)
        self.electrode_type = electrode_type
        self.dt = dt

    def current_spread(
        self,
        positions: torch.Tensor,
        amplitude: float,
        tissue_impedance: float = 1000.0  # Ω
    ) -> torch.Tensor:
        """
        Compute current spread from electrode.

        V(r) = I * ρ / (4π * r)  (point source approximation)

        Args:
            positions: Neuron positions (n_neurons, 3)
            amplitude: Stimulation current (mA)
            tissue_impedance: Tissue impedance (Ω)

        Returns:
            Voltage at each position (mV)
        """
        distance = torch.norm(positions - self.electrode_position, dim=1)

        # Point source approximation
        resistivity = tissue_impedance * 1e-2  # Ω·m
        voltage = (amplitude * resistivity) / (4 * np.pi * (distance + 0.1))  # mV

        return voltage

    def activation_function(
        self,
        positions: torch.Tensor,
        voltage: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute axon activation from applied voltage.

        Uses activating function: f = -dV/dx

        Args:
            positions: Neuron positions (n_neurons, 3)
            voltage: Applied voltage (mV)

        Returns:
            Activation likelihood [0, 1]
        """
        # Simplified: higher voltage → more activation
        activation_threshold = 10.0  # mV
        activation = torch.sigmoid((voltage - activation_threshold) / 2.0)

        return activation

    def forward(
        self,
        positions: torch.Tensor,
        params: StimulationParameters,
        duration: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Simulate DBS.

        Args:
            positions: Neuron positions (n_neurons, 3)
            params: Stimulation parameters
            duration: Duration (ms)

        Returns:
            (current_waveform, neuron_activation): Stimulation and response
        """
        n_steps = int(duration / self.dt)
        n_neurons = positions.shape[0]

        time = torch.arange(n_steps, dtype=torch.float32) * self.dt
        current = torch.zeros(n_steps)
        activation = torch.zeros(n_steps, n_neurons)

        # Generate pulse train
        pulse_period = 1000.0 / params.frequency  # ms
        n_pulses_in_duration = int(duration / pulse_period)

        for pulse_idx in range(n_pulses_in_duration):
            pulse_start = pulse_idx * pulse_period
            pulse_end = pulse_start + params.pulse_width

            start_idx = int(pulse_start / self.dt)
            end_idx = int(pulse_end / self.dt)

            if end_idx < n_steps:
                # Biphasic pulse
                if params.waveform == 'biphasic':
                    mid_idx = (start_idx + end_idx) // 2
                    current[start_idx:mid_idx] = params.amplitude
                    current[mid_idx:end_idx] = -params.amplitude
                elif params.waveform == 'monophasic':
                    current[start_idx:end_idx] = params.amplitude

                # Compute activation
                for t_idx in range(start_idx, end_idx):
                    voltage = self.current_spread(positions, current[t_idx])
                    activation[t_idx] = self.activation_function(positions, voltage)

        return current, activation


class ElectricalStimulation(nn.Module):
    """
    Direct electrical stimulation.

    Intracranial microstimulation for precise circuit manipulation.
    """

    def __init__(
        self,
        electrode_positions: torch.Tensor,  # (n_electrodes, 3)
        dt: float = 0.1
    ):
        super().__init__()
        self.register_buffer('electrode_positions', electrode_positions)
        self.dt = dt
        self.n_electrodes = electrode_positions.shape[0]

    def stimulate(
        self,
        neuron_positions: torch.Tensor,
        electrode_idx: int,
        amplitude: float,
        duration: float
    ) -> torch.Tensor:
        """
        Apply electrical stimulation from one electrode.

        Args:
            neuron_positions: Neuron positions (n_neurons, 3)
            electrode_idx: Which electrode to use
            amplitude: Current amplitude (μA)
            duration: Pulse duration (ms)

        Returns:
            Injected current at each neuron (n_neurons,)
        """
        electrode_pos = self.electrode_positions[electrode_idx]

        # Distance-dependent current spread
        distance = torch.norm(neuron_positions - electrode_pos, dim=1)

        # Exponential decay with distance
        space_constant = 100.0  # μm
        current = amplitude * torch.exp(-distance / space_constant)

        return current


class TDCS(nn.Module):
    """
    Transcranial Direct Current Stimulation (tDCS).

    Low-intensity constant current to modulate cortical excitability.
    """

    def __init__(
        self,
        anode_position: torch.Tensor,
        cathode_position: torch.Tensor,
        current_density: float = 0.05  # mA/cm^2
    ):
        super().__init__()
        self.register_buffer('anode_position', anode_position)
        self.register_buffer('cathode_position', cathode_position)
        self.current_density = current_density

    def forward(
        self,
        neuron_positions: torch.Tensor,
        polarity: float,  # +1 for anodal, -1 for cathodal
        duration: float
    ) -> torch.Tensor:
        """
        Compute tDCS-induced polarization.

        Args:
            neuron_positions: Neuron positions (n_neurons, 3)
            polarity: Stimulation polarity
            duration: Duration (s)

        Returns:
            Membrane polarization (mV)
        """
        # Simplified: uniform field between electrodes
        electrode_vector = self.cathode_position - self.anode_position
        electrode_distance = torch.norm(electrode_vector)

        # Project neurons onto electrode axis
        neuron_vectors = neuron_positions - self.anode_position
        projection = torch.sum(
            neuron_vectors * electrode_vector / electrode_distance,
            dim=1
        ) / electrode_distance

        # Linear voltage drop
        max_voltage = self.current_density * 1000.0 * duration / 1000.0  # mV
        polarization = polarity * max_voltage * (0.5 - projection)

        return polarization


class StimulationExperiment:
    """
    Design and analyze neural stimulation experiments.
    """

    @staticmethod
    def theta_burst_stimulation(
        frequency: float = 50.0,  # Hz (gamma)
        burst_frequency: float = 5.0,  # Hz (theta)
        n_pulses_per_burst: int = 3,
        n_bursts: int = 10
    ) -> StimulationParameters:
        """
        Create theta-burst stimulation (TBS) protocol.

        Args:
            frequency: Pulse frequency within burst (Hz)
            burst_frequency: Burst repetition rate (Hz)
            n_pulses_per_burst: Pulses per burst
            n_bursts: Number of bursts

        Returns:
            Stimulation parameters
        """
        pulse_width = 1.0  # ms (typical for TMS)
        interpulse_interval = 1000.0 / frequency

        return StimulationParameters(
            amplitude=0.8,
            frequency=frequency,
            pulse_width=pulse_width,
            n_pulses=n_pulses_per_burst * n_bursts,
            interpulse_interval=interpulse_interval,
            waveform='biphasic'
        )

    @staticmethod
    def paired_pulse_paradigm(
        interstimulus_interval: float,  # ms
        amplitude1: float = 1.0,
        amplitude2: float = 1.0
    ) -> List[StimulationParameters]:
        """
        Create paired-pulse stimulation for testing inhibition/facilitation.

        Args:
            interstimulus_interval: ISI between pulses (ms)
            amplitude1: First pulse amplitude
            amplitude2: Second pulse (conditioning pulse) amplitude

        Returns:
            List of two stimulation parameters
        """
        pulse1 = StimulationParameters(
            amplitude=amplitude1,
            frequency=0,  # Single pulse
            pulse_width=1.0,
            n_pulses=1,
            interpulse_interval=0
        )

        pulse2 = StimulationParameters(
            amplitude=amplitude2,
            frequency=0,
            pulse_width=1.0,
            n_pulses=1,
            interpulse_interval=interstimulus_interval
        )

        return [pulse1, pulse2]

    @staticmethod
    def high_frequency_stimulation(
        frequency: float = 130.0,  # Hz (typical for PD)
        amplitude: float = 3.0,  # mA
        pulse_width: float = 0.06  # ms (60 μs)
    ) -> StimulationParameters:
        """
        High-frequency DBS (e.g., for Parkinson's disease).

        Args:
            frequency: Stimulation frequency (Hz)
            amplitude: Current amplitude (mA)
            pulse_width: Pulse duration (ms)

        Returns:
            DBS parameters
        """
        return StimulationParameters(
            amplitude=amplitude,
            frequency=frequency,
            pulse_width=pulse_width,
            n_pulses=int(frequency * 10),  # 10 seconds
            interpulse_interval=1000.0 / frequency,
            waveform='biphasic'
        )
