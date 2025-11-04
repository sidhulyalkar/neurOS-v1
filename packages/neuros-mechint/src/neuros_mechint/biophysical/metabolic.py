"""
Metabolic Constraints and Energy Dynamics

Models ATP production, consumption, and energy efficiency constraints
for biologically realistic neural computation.

References:
- Attwell & Laughlin (2001): An energy budget for signaling
- Alle et al. (2009): Energy-efficient action potentials
- Harris et al. (2012): Synaptic energy use and supply
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class EnergyBudget:
    """Energy costs for neural operations."""
    # ATP molecules per event
    action_potential: float = 1.64e8  # ATP/spike
    synaptic_transmission: float = 1.6e5  # ATP/vesicle
    resting_potential: float = 3.4e8  # ATP/s/neuron
    postsynaptic_current: float = 8.5e4  # ATP/synaptic event

    # Conversion factors
    atp_per_glucose: float = 30.0  # ATP from one glucose
    glucose_molecular_weight: float = 180.0  # g/mol


class ATPDynamics(nn.Module):
    """
    ATP production and consumption dynamics.

    Models cellular energy metabolism with glycolysis
    and oxidative phosphorylation.
    """

    def __init__(
        self,
        ATP_rest: float = 2.5,  # mM (resting ATP concentration)
        ADP_rest: float = 0.01,  # mM
        glucose: float = 5.0,    # mM (brain glucose)
        production_rate: float = 1.0,  # Relative rate
        dt: float = 1.0  # ms
    ):
        super().__init__()
        self.ATP_rest = ATP_rest
        self.ADP_rest = ADP_rest
        self.glucose = glucose
        self.production_rate = production_rate
        self.dt = dt

        # State variables
        self.register_buffer('ATP', None)
        self.register_buffer('ADP', None)
        self.register_buffer('Pi', None)  # Inorganic phosphate

    def glycolysis_rate(self, glucose: torch.Tensor, ATP: torch.Tensor) -> torch.Tensor:
        """
        Glycolysis ATP production rate (Michaelis-Menten).

        Args:
            glucose: Glucose concentration (mM)
            ATP: ATP concentration (mM)

        Returns:
            Production rate (mM/ms)
        """
        V_max = 0.05  # mM/ms
        K_m = 0.1  # mM
        K_i = 5.0  # mM (ATP inhibition)

        rate = V_max * glucose / (K_m + glucose) * (1.0 / (1.0 + ATP / K_i))
        return rate * self.production_rate

    def oxidative_phosphorylation(
        self,
        ADP: torch.Tensor,
        Pi: torch.Tensor,
        O2: float = 0.1  # mM
    ) -> torch.Tensor:
        """
        Oxidative phosphorylation rate.

        Args:
            ADP: ADP concentration (mM)
            Pi: Phosphate concentration (mM)
            O2: Oxygen concentration (mM)

        Returns:
            ATP production rate (mM/ms)
        """
        V_max = 0.2  # mM/ms
        K_ADP = 0.03  # mM
        K_Pi = 1.0  # mM

        rate = V_max * (ADP / (K_ADP + ADP)) * (Pi / (K_Pi + Pi))
        return rate * self.production_rate

    def consume_ATP(self, consumption: torch.Tensor):
        """
        Consume ATP for neural activity.

        Args:
            consumption: ATP consumption (mM)
        """
        if self.ATP is None:
            raise RuntimeError("ATP dynamics not initialized")

        self.ATP = self.ATP - consumption
        self.ADP = self.ADP + consumption
        self.Pi = self.Pi + consumption

        # Prevent negative values
        self.ATP = torch.clamp(self.ATP, 0.0, None)

    def step(self, glucose: Optional[torch.Tensor] = None):
        """
        Single time step of ATP dynamics.

        Args:
            glucose: Glucose concentration (mM), uses default if None
        """
        if self.ATP is None:
            raise RuntimeError("Call initialize() first")

        if glucose is None:
            glucose = torch.full_like(self.ATP, self.glucose)

        # Production
        glycolysis = self.glycolysis_rate(glucose, self.ATP)
        ox_phos = self.oxidative_phosphorylation(self.ADP, self.Pi)

        production = glycolysis + ox_phos

        # Update concentrations
        self.ATP = self.ATP + self.dt * production
        self.ADP = self.ADP - self.dt * production
        self.Pi = self.Pi - self.dt * production

        # Conservation
        total = self.ATP + self.ADP
        target_total = self.ATP_rest + self.ADP_rest

        # Slow equilibration
        self.ATP = self.ATP + 0.001 * (target_total - total) * (self.ATP / total)
        self.ADP = self.ADP + 0.001 * (target_total - total) * (self.ADP / total)

    def initialize(self, batch_size: int, device: str = 'cpu'):
        """Initialize ATP concentrations."""
        self.ATP = torch.full((batch_size,), self.ATP_rest, device=device)
        self.ADP = torch.full((batch_size,), self.ADP_rest, device=device)
        self.Pi = torch.full((batch_size,), 1.0, device=device)

    def reset(self):
        """Reset to resting levels."""
        if self.ATP is not None:
            self.ATP.fill_(self.ATP_rest)
            self.ADP.fill_(self.ADP_rest)
            self.Pi.fill_(1.0)


class MetabolicConstraint(nn.Module):
    """
    Enforce metabolic constraints on neural network.

    Limits activity based on available ATP and energy budget.
    """

    def __init__(
        self,
        n_neurons: int,
        energy_budget: Optional[EnergyBudget] = None,
        ATP_threshold: float = 0.5,  # Minimum ATP (mM)
        dt: float = 1.0
    ):
        super().__init__()
        self.n_neurons = n_neurons
        self.energy_budget = energy_budget or EnergyBudget()
        self.ATP_threshold = ATP_threshold

        self.atp_dynamics = ATPDynamics(dt=dt)

    def compute_spike_cost(self, spikes: torch.Tensor) -> torch.Tensor:
        """
        Compute ATP cost of action potentials.

        Args:
            spikes: Spike train (batch, n_neurons)

        Returns:
            ATP consumed (mM)
        """
        # Convert molecule count to mM
        # Assuming neuron volume ~ 10^-12 L
        volume_L = 1e-12
        avogadro = 6.02e23

        atp_mol = self.energy_budget.action_potential * spikes / avogadro
        atp_mM = (atp_mol / volume_L) * 1000  # Convert to mM

        return atp_mM

    def compute_synaptic_cost(
        self,
        n_synapses_active: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute ATP cost of synaptic transmission.

        Args:
            n_synapses_active: Number of active synapses

        Returns:
            ATP consumed (mM)
        """
        volume_L = 1e-12
        avogadro = 6.02e23

        atp_mol = self.energy_budget.synaptic_transmission * n_synapses_active / avogadro
        atp_mM = (atp_mol / volume_L) * 1000

        return atp_mM

    def forward(
        self,
        spikes: torch.Tensor,
        synaptic_activity: torch.Tensor,
        enforce: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Apply metabolic constraints.

        Args:
            spikes: Spike train (batch, n_neurons)
            synaptic_activity: Synaptic events (batch, n_neurons)
            enforce: Whether to suppress activity when ATP low

        Returns:
            (constrained_spikes, metrics): Modified spikes and energy metrics
        """
        batch_size = spikes.shape[0]
        device = spikes.device

        if self.atp_dynamics.ATP is None:
            self.atp_dynamics.initialize(batch_size, device)

        # Compute costs
        spike_cost = self.compute_spike_cost(spikes)
        synaptic_cost = self.compute_synaptic_cost(synaptic_activity)

        total_cost = spike_cost + synaptic_cost

        # Consume ATP
        self.atp_dynamics.consume_ATP(total_cost.mean(1))

        # Update ATP production
        self.atp_dynamics.step()

        # Enforce constraint
        if enforce:
            # Suppress activity if ATP too low
            atp_available = self.atp_dynamics.ATP > self.ATP_threshold
            constrained_spikes = spikes * atp_available.unsqueeze(1).float()
        else:
            constrained_spikes = spikes

        metrics = {
            'ATP': self.atp_dynamics.ATP.clone(),
            'ADP': self.atp_dynamics.ADP.clone(),
            'spike_cost': spike_cost,
            'synaptic_cost': synaptic_cost,
            'total_cost': total_cost
        }

        return constrained_spikes, metrics


class EnergyEfficiencyAnalyzer:
    """
    Analyze energy efficiency of neural computations.

    Measures:
    - Bits per ATP
    - Spikes per bit
    - Energy per synaptic event
    - Metabolic rate
    """

    def __init__(self, energy_budget: Optional[EnergyBudget] = None):
        self.energy_budget = energy_budget or EnergyBudget()

    def bits_per_joule(
        self,
        information_bits: float,
        atp_consumed: float
    ) -> float:
        """
        Compute computational efficiency.

        Args:
            information_bits: Information processed (bits)
            atp_consumed: ATP molecules consumed

        Returns:
            Bits per Joule
        """
        # ATP hydrolysis: ~30.5 kJ/mol
        atp_energy_J = (atp_consumed / 6.02e23) * 30500

        if atp_energy_J == 0:
            return 0.0

        return information_bits / atp_energy_J

    def metabolic_rate(
        self,
        atp_consumed: float,
        time_s: float
    ) -> float:
        """
        Compute metabolic rate.

        Args:
            atp_consumed: ATP molecules
            time_s: Time (seconds)

        Returns:
            Watts
        """
        # ATP to Joules
        atp_energy_J = (atp_consumed / 6.02e23) * 30500

        return atp_energy_J / time_s

    def compare_to_brain(
        self,
        power_watts: float,
        n_neurons: int
    ) -> Dict[str, float]:
        """
        Compare to human brain efficiency.

        Args:
            power_watts: Power consumption
            n_neurons: Number of neurons

        Returns:
            Comparison metrics
        """
        # Human brain: ~20W for ~86 billion neurons
        brain_power_per_neuron = 20.0 / 86e9

        return {
            'power_per_neuron': power_watts / n_neurons,
            'brain_power_per_neuron': brain_power_per_neuron,
            'efficiency_ratio': (power_watts / n_neurons) / brain_power_per_neuron,
            'total_power': power_watts,
            'n_neurons': n_neurons
        }

    def wiring_cost(
        self,
        connectivity_matrix: torch.Tensor,
        positions: Optional[torch.Tensor] = None
    ) -> float:
        """
        Estimate wiring cost (physical distance × connections).

        Args:
            connectivity_matrix: (n, n) connection weights
            positions: (n, 3) neuron positions

        Returns:
            Total wiring cost
        """
        n = connectivity_matrix.shape[0]

        if positions is None:
            # Assume random 3D positions
            positions = torch.randn(n, 3)

        # Pairwise distances
        dist_matrix = torch.cdist(positions, positions)

        # Wiring cost: sum of (distance × connection strength)
        wiring_cost = (dist_matrix * connectivity_matrix.abs()).sum()

        return wiring_cost.item()


class GlucoseTransport(nn.Module):
    """
    Model glucose transport from blood to neurons.

    Includes blood-brain barrier transport and astrocyte-neuron
    lactate shuttle.
    """

    def __init__(
        self,
        blood_glucose: float = 5.0,  # mM
        K_t: float = 1.0,  # mM (transporter affinity)
        V_max: float = 1.0,  # mM/ms
        dt: float = 1.0
    ):
        super().__init__()
        self.blood_glucose = blood_glucose
        self.K_t = K_t
        self.V_max = V_max
        self.dt = dt

        self.register_buffer('brain_glucose', None)

    def transport_rate(self, blood_glucose: torch.Tensor) -> torch.Tensor:
        """
        GLUT1 transporter rate (Michaelis-Menten).

        Args:
            blood_glucose: Blood glucose concentration (mM)

        Returns:
            Transport rate (mM/ms)
        """
        return self.V_max * blood_glucose / (self.K_t + blood_glucose)

    def forward(
        self,
        consumption_rate: torch.Tensor,
        blood_glucose: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Update brain glucose based on transport and consumption.

        Args:
            consumption_rate: Glucose consumption (mM/ms)
            blood_glucose: Blood glucose (mM)

        Returns:
            Current brain glucose (mM)
        """
        batch_size = consumption_rate.shape[0]
        device = consumption_rate.device

        if self.brain_glucose is None or self.brain_glucose.shape[0] != batch_size:
            self.brain_glucose = torch.full((batch_size,), 2.5, device=device)

        if blood_glucose is None:
            blood_glucose = torch.full_like(self.brain_glucose, self.blood_glucose)

        # Transport from blood
        transport = self.transport_rate(blood_glucose)

        # Update brain glucose
        dG = transport - consumption_rate
        self.brain_glucose = self.brain_glucose + self.dt * dG

        self.brain_glucose = torch.clamp(self.brain_glucose, 0.0, None)

        return self.brain_glucose
