"""
Pharmacological Interventions

Models drug effects on neurotransmitter systems, receptors, and ion channels.
Includes agonists, antagonists, channel blockers, and enzyme modulators.

References:
- Cooper et al. (2003): The Biochemical Basis of Neuropharmacology
- Stahl (2013): Stahl's Essential Psychopharmacology
- Rang & Dale (2015): Pharmacology
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List
from dataclasses import dataclass
from enum import Enum


class DrugMechanism(Enum):
    """Mechanism of drug action."""
    AGONIST = "agonist"  # Activates receptor
    ANTAGONIST = "antagonist"  # Blocks receptor
    PAM = "positive_allosteric_modulator"  # Enhances receptor
    NAM = "negative_allosteric_modulator"  # Reduces receptor
    CHANNEL_BLOCKER = "channel_blocker"  # Blocks ion channel
    REUPTAKE_INHIBITOR = "reuptake_inhibitor"  # Blocks transporter
    ENZYME_INHIBITOR = "enzyme_inhibitor"  # Inhibits metabolic enzyme
    ENZYME_ENHANCER = "enzyme_enhancer"  # Enhances enzyme


@dataclass
class DrugParameters:
    """Pharmacological parameters for a drug."""
    EC50: float  # Half-maximal effective concentration (μM)
    IC50: float  # Half-maximal inhibitory concentration (μM)
    Ki: float  # Inhibition constant (μM)
    efficacy: float  # Maximal efficacy [0, 1]
    hill_coefficient: float = 1.0  # Hill coefficient for cooperativity
    onset_tau: float = 100.0  # Time to reach effect (ms)
    offset_tau: float = 1000.0  # Time to washout (ms)


class Drug(nn.Module):
    """Base class for pharmacological agents."""

    def __init__(
        self,
        name: str,
        mechanism: DrugMechanism,
        params: DrugParameters,
        target: str,  # Receptor/channel type
        dt: float = 1.0
    ):
        super().__init__()
        self.name = name
        self.mechanism = mechanism
        self.params = params
        self.target = target
        self.dt = dt

        # Drug state
        self.register_buffer('concentration', None)
        self.register_buffer('bound_fraction', None)

    def dose_response(self, concentration: torch.Tensor) -> torch.Tensor:
        """
        Hill equation for dose-response.

        E = E_max * [D]^n / (EC50^n + [D]^n)

        Args:
            concentration: Drug concentration (μM)

        Returns:
            Effect [0, 1]
        """
        n = self.params.hill_coefficient
        if self.mechanism in [DrugMechanism.AGONIST, DrugMechanism.PAM]:
            EC50 = self.params.EC50
            effect = self.params.efficacy * (concentration ** n) / (EC50 ** n + concentration ** n)
        else:  # Antagonist, NAM, blockers
            IC50 = self.params.IC50
            effect = 1.0 - (concentration ** n) / (IC50 ** n + concentration ** n)

        return effect

    def update_concentration(
        self,
        applied_dose: torch.Tensor,
        metabolism_rate: float = 0.01
    ):
        """
        Update drug concentration with application and metabolism.

        Args:
            applied_dose: Newly applied dose (μM)
            metabolism_rate: Clearance rate (1/ms)
        """
        batch_size = applied_dose.shape[0]
        device = applied_dose.device

        if self.concentration is None or self.concentration.shape[0] != batch_size:
            self.concentration = torch.zeros(batch_size, device=device)
            self.bound_fraction = torch.zeros(batch_size, device=device)

        # Absorption
        dC_absorption = applied_dose / self.params.onset_tau

        # Metabolism/clearance
        dC_metabolism = -metabolism_rate * self.concentration

        # Update
        self.concentration = self.concentration + self.dt * (dC_absorption + dC_metabolism)
        self.concentration = torch.clamp(self.concentration, 0.0, None)

        # Binding dynamics
        effect = self.dose_response(self.concentration)
        dB = (effect - self.bound_fraction) / self.params.onset_tau
        self.bound_fraction = self.bound_fraction + self.dt * dB

    def modulate_conductance(
        self,
        baseline_conductance: torch.Tensor
    ) -> torch.Tensor:
        """
        Modulate ion channel conductance.

        Args:
            baseline_conductance: Baseline conductance (nS)

        Returns:
            Modulated conductance
        """
        if self.concentration is None:
            return baseline_conductance

        if self.mechanism == DrugMechanism.CHANNEL_BLOCKER:
            # Reduce conductance
            return baseline_conductance * (1.0 - self.bound_fraction)
        elif self.mechanism == DrugMechanism.PAM:
            # Enhance conductance
            return baseline_conductance * (1.0 + self.bound_fraction)
        elif self.mechanism == DrugMechanism.NAM:
            # Reduce conductance
            return baseline_conductance * (1.0 - 0.5 * self.bound_fraction)
        else:
            return baseline_conductance

    def modulate_receptor(
        self,
        baseline_response: torch.Tensor
    ) -> torch.Tensor:
        """
        Modulate receptor response.

        Args:
            baseline_response: Baseline receptor activation

        Returns:
            Modulated response
        """
        if self.concentration is None:
            return baseline_response

        if self.mechanism == DrugMechanism.AGONIST:
            # Add agonist effect
            return baseline_response + self.bound_fraction * self.params.efficacy
        elif self.mechanism == DrugMechanism.ANTAGONIST:
            # Block receptor
            return baseline_response * (1.0 - self.bound_fraction)
        elif self.mechanism == DrugMechanism.PAM:
            # Enhance response
            return baseline_response * (1.0 + self.bound_fraction)
        else:
            return baseline_response

    def reset(self):
        """Reset drug state."""
        if self.concentration is not None:
            self.concentration.zero_()
            self.bound_fraction.zero_()


# ==================== Common Drugs ====================

class Drugs:
    """Factory for common neuropharmacological agents."""

    @staticmethod
    def TTX() -> Drug:
        """Tetrodotoxin - sodium channel blocker."""
        return Drug(
            name="TTX",
            mechanism=DrugMechanism.CHANNEL_BLOCKER,
            params=DrugParameters(
                EC50=1e-3, IC50=0.001, Ki=0.001, efficacy=1.0,
                onset_tau=50.0, offset_tau=500.0
            ),
            target="Na_channel"
        )

    @staticmethod
    def TEA() -> Drug:
        """Tetraethylammonium - potassium channel blocker."""
        return Drug(
            name="TEA",
            mechanism=DrugMechanism.CHANNEL_BLOCKER,
            params=DrugParameters(
                EC50=1.0, IC50=0.5, Ki=0.5, efficacy=1.0,
                onset_tau=100.0, offset_tau=1000.0
            ),
            target="K_channel"
        )

    @staticmethod
    def APV() -> Drug:
        """APV/AP5 - NMDA receptor antagonist."""
        return Drug(
            name="APV",
            mechanism=DrugMechanism.ANTAGONIST,
            params=DrugParameters(
                EC50=10.0, IC50=0.5, Ki=0.5, efficacy=1.0,
                onset_tau=200.0, offset_tau=2000.0
            ),
            target="NMDA"
        )

    @staticmethod
    def CNQX() -> Drug:
        """CNQX - AMPA/kainate receptor antagonist."""
        return Drug(
            name="CNQX",
            mechanism=DrugMechanism.ANTAGONIST,
            params=DrugParameters(
                EC50=10.0, IC50=0.3, Ki=0.3, efficacy=1.0,
                onset_tau=100.0, offset_tau=1000.0
            ),
            target="AMPA"
        )

    @staticmethod
    def Bicuculline() -> Drug:
        """Bicuculline - GABA-A receptor antagonist."""
        return Drug(
            name="Bicuculline",
            mechanism=DrugMechanism.ANTAGONIST,
            params=DrugParameters(
                EC50=10.0, IC50=1.0, Ki=1.0, efficacy=1.0,
                onset_tau=150.0, offset_tau=1500.0
            ),
            target="GABA_A"
        )

    @staticmethod
    def Muscimol() -> Drug:
        """Muscimol - GABA-A receptor agonist."""
        return Drug(
            name="Muscimol",
            mechanism=DrugMechanism.AGONIST,
            params=DrugParameters(
                EC50=1.0, IC50=10.0, Ki=1.0, efficacy=0.9,
                onset_tau=200.0, offset_tau=2000.0
            ),
            target="GABA_A"
        )

    @staticmethod
    def Baclofen() -> Drug:
        """Baclofen - GABA-B receptor agonist."""
        return Drug(
            name="Baclofen",
            mechanism=DrugMechanism.AGONIST,
            params=DrugParameters(
                EC50=5.0, IC50=10.0, Ki=5.0, efficacy=0.8,
                onset_tau=500.0, offset_tau=5000.0
            ),
            target="GABA_B"
        )

    @staticmethod
    def Scopolamine() -> Drug:
        """Scopolamine - muscarinic acetylcholine receptor antagonist."""
        return Drug(
            name="Scopolamine",
            mechanism=DrugMechanism.ANTAGONIST,
            params=DrugParameters(
                EC50=10.0, IC50=0.1, Ki=0.1, efficacy=1.0,
                onset_tau=300.0, offset_tau=3000.0
            ),
            target="mAChR"
        )

    @staticmethod
    def Ketamine() -> Drug:
        """Ketamine - NMDA receptor antagonist (non-competitive)."""
        return Drug(
            name="Ketamine",
            mechanism=DrugMechanism.ANTAGONIST,
            params=DrugParameters(
                EC50=10.0, IC50=2.0, Ki=2.0, efficacy=0.7,
                onset_tau=500.0, offset_tau=10000.0
            ),
            target="NMDA"
        )

    @staticmethod
    def Fluoxetine() -> Drug:
        """Fluoxetine (Prozac) - serotonin reuptake inhibitor."""
        return Drug(
            name="Fluoxetine",
            mechanism=DrugMechanism.REUPTAKE_INHIBITOR,
            params=DrugParameters(
                EC50=5.0, IC50=0.01, Ki=0.01, efficacy=0.9,
                onset_tau=10000.0, offset_tau=100000.0
            ),
            target="SERT"
        )

    @staticmethod
    def Methylphenidate() -> Drug:
        """Methylphenidate (Ritalin) - dopamine/norepinephrine reuptake inhibitor."""
        return Drug(
            name="Methylphenidate",
            mechanism=DrugMechanism.REUPTAKE_INHIBITOR,
            params=DrugParameters(
                EC50=10.0, IC50=0.2, Ki=0.2, efficacy=0.8,
                onset_tau=1000.0, offset_tau=5000.0
            ),
            target="DAT"
        )


class PharmacologyExperiment:
    """
    Design and analyze pharmacological experiments.

    Supports:
    - Dose-response curves
    - Drug combination effects
    - Receptor occupancy
    - Time-course analysis
    """

    def __init__(self):
        pass

    def dose_response_curve(
        self,
        drug: Drug,
        concentrations: List[float],
        response_fn: callable
    ) -> Dict[str, torch.Tensor]:
        """
        Measure dose-response relationship.

        Args:
            drug: Drug to test
            concentrations: List of concentrations (μM)
            response_fn: Function that measures response

        Returns:
            Dictionary with concentrations and responses
        """
        concs = torch.tensor(concentrations, dtype=torch.float32)
        responses = torch.zeros_like(concs)

        for i, conc in enumerate(concs):
            drug.concentration = conc.unsqueeze(0)
            responses[i] = response_fn(drug)

        drug.reset()

        return {
            'concentrations': concs,
            'responses': responses
        }

    def receptor_occupancy(
        self,
        drug: Drug,
        concentration: float
    ) -> float:
        """
        Calculate receptor occupancy (Langmuir equation).

        θ = [D] / (Kd + [D])

        Args:
            drug: Drug
            concentration: Drug concentration (μM)

        Returns:
            Occupancy [0, 1]
        """
        Kd = drug.params.Ki
        return concentration / (Kd + concentration)

    def drug_combination(
        self,
        drug1: Drug,
        drug2: Drug,
        dose1: float,
        dose2: float,
        interaction: str = 'additive'
    ) -> float:
        """
        Model drug combination effects.

        Args:
            drug1: First drug
            drug2: Second drug
            dose1: Dose of drug 1 (μM)
            dose2: Dose of drug 2 (μM)
            interaction: 'additive', 'synergistic', or 'antagonistic'

        Returns:
            Combined effect
        """
        effect1 = drug1.dose_response(torch.tensor([dose1]))[0].item()
        effect2 = drug2.dose_response(torch.tensor([dose2]))[0].item()

        if interaction == 'additive':
            return effect1 + effect2
        elif interaction == 'synergistic':
            # Greater than additive
            return effect1 + effect2 + 0.3 * effect1 * effect2
        elif interaction == 'antagonistic':
            # Less than additive
            return effect1 + effect2 - 0.5 * effect1 * effect2
        else:
            raise ValueError(f"Unknown interaction: {interaction}")

    def time_course(
        self,
        drug: Drug,
        applied_dose: float,
        duration: float,
        dt: float = 1.0,
        metabolism_rate: float = 0.001
    ) -> Dict[str, torch.Tensor]:
        """
        Simulate time course of drug effects.

        Args:
            drug: Drug
            applied_dose: Applied dose (μM)
            duration: Duration (ms)
            dt: Time step (ms)
            metabolism_rate: Clearance rate (1/ms)

        Returns:
            Time course of concentration and effect
        """
        n_steps = int(duration / dt)
        time = torch.arange(n_steps, dtype=torch.float32) * dt
        concentration = torch.zeros(n_steps)
        effect = torch.zeros(n_steps)

        drug.reset()
        dose_tensor = torch.tensor([applied_dose])

        for t in range(n_steps):
            # Apply dose only at t=0
            dose_at_t = dose_tensor if t == 0 else torch.zeros(1)
            drug.update_concentration(dose_at_t, metabolism_rate)

            concentration[t] = drug.concentration[0]
            effect[t] = drug.bound_fraction[0]

        drug.reset()

        return {
            'time': time,
            'concentration': concentration,
            'effect': effect
        }
