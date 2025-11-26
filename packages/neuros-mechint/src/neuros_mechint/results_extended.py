"""
Extended Result Types for Advanced Mechanistic Interpretability Analyses

Specialized result classes for:
- Biophysical modeling (ion channels, compartments, metabolism)
- Interventions (optogenetics, pharmacology, stimulation)
- Criticality analysis (avalanches, branching processes)
- Multifractal analysis (MF-DFA, WTMM)
- Temporal dynamics (ISC, TRF, phase analysis)

These extend the base MechIntResult class with domain-specific attributes
and methods for specialized analyses.

Author: NeuroS Team
Date: 2025-11-04
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import h5py
import json

from neuros_mechint.results import MechIntResult


@dataclass
class BiophysicalResult(MechIntResult):
    """
    Results from biophysical neural modeling.

    Stores voltages, currents, conductances, and metabolic state
    from compartmental or single-neuron simulations.
    """

    # Electrical dynamics
    voltages: Optional[np.ndarray] = None  # Membrane potentials over time
    currents: Optional[Dict[str, np.ndarray]] = None  # Ion channel currents
    conductances: Optional[Dict[str, np.ndarray]] = None  # Channel conductances

    # Synaptic dynamics
    synaptic_weights: Optional[np.ndarray] = None  # Synaptic weight matrix
    plasticity_traces: Optional[Dict[str, np.ndarray]] = None  # STDP/STP traces

    # Metabolic state
    atp_levels: Optional[np.ndarray] = None  # ATP concentration over time
    glucose_levels: Optional[np.ndarray] = None  # Glucose levels
    oxygen_consumption: Optional[np.ndarray] = None  # O2 consumption rate

    # Spike analysis
    spike_times: Optional[List[np.ndarray]] = None  # Spike times per neuron
    spike_rate: Optional[float] = None  # Mean spike rate (Hz)
    isi_distribution: Optional[np.ndarray] = None  # Inter-spike intervals

    # Compartment-specific (if multi-compartment)
    compartment_names: Optional[List[str]] = None
    compartment_voltages: Optional[Dict[str, np.ndarray]] = None

    def save(self, filepath: str):
        """Save with specialized biophysical data."""
        with h5py.File(filepath, 'w') as f:
            # Base data (manually save instead of calling non-existent method)
            f.attrs['method'] = self.method
            f.attrs['timestamp'] = self.timestamp if self.timestamp else ''
            f.attrs['content_hash'] = self.content_hash if self.content_hash else ''
            f.attrs['metadata'] = json.dumps(self.metadata)
            f.attrs['metrics'] = json.dumps(self.metrics)

            # Save base data
            if isinstance(self.data, dict):
                base_data_group = f.create_group('base_data')
                for key, value in self.data.items():
                    if isinstance(value, (np.ndarray, list)):
                        arr = np.array(value)
                        base_data_group.create_dataset(key, data=arr, compression='gzip')
                    else:
                        base_data_group.attrs[key] = str(value)

            # Electrical dynamics
            if self.voltages is not None:
                f.create_dataset('voltages', data=self.voltages, compression='gzip')

            if self.currents:
                currents_group = f.create_group('currents')
                for name, current in self.currents.items():
                    currents_group.create_dataset(name, data=current, compression='gzip')

            if self.conductances:
                cond_group = f.create_group('conductances')
                for name, cond in self.conductances.items():
                    cond_group.create_dataset(name, data=cond, compression='gzip')

            # Synaptic
            if self.synaptic_weights is not None:
                f.create_dataset('synaptic_weights', data=self.synaptic_weights, compression='gzip')

            if self.plasticity_traces:
                plast_group = f.create_group('plasticity_traces')
                for name, trace in self.plasticity_traces.items():
                    plast_group.create_dataset(name, data=trace, compression='gzip')

            # Metabolic
            if self.atp_levels is not None:
                f.create_dataset('atp_levels', data=self.atp_levels, compression='gzip')

            if self.glucose_levels is not None:
                f.create_dataset('glucose_levels', data=self.glucose_levels, compression='gzip')

            if self.oxygen_consumption is not None:
                f.create_dataset('oxygen_consumption', data=self.oxygen_consumption, compression='gzip')

            # Spikes
            if self.spike_times:
                spikes_group = f.create_group('spike_times')
                for i, times in enumerate(self.spike_times):
                    spikes_group.create_dataset(f'neuron_{i}', data=times)

            if self.isi_distribution is not None:
                f.create_dataset('isi_distribution', data=self.isi_distribution)

            # Compartments
            if self.compartment_voltages:
                comp_group = f.create_group('compartment_voltages')
                for name, voltage in self.compartment_voltages.items():
                    comp_group.create_dataset(name, data=voltage, compression='gzip')

            # Metadata
            f.attrs['spike_rate'] = self.spike_rate if self.spike_rate else 0.0
            if self.compartment_names:
                f.attrs['compartment_names'] = ','.join(self.compartment_names)

    @classmethod
    def load(cls, filepath: str) -> 'BiophysicalResult':
        """Load biophysical result from HDF5."""
        with h5py.File(filepath, 'r') as f:
            # Load base data manually
            method = f.attrs['method']
            metadata = json.loads(f.attrs['metadata'])
            metrics = json.loads(f.attrs['metrics'])
            timestamp = f.attrs.get('timestamp', '')
            content_hash = f.attrs.get('content_hash', '')

            # Load base data dict
            data = {}
            if 'base_data' in f:
                for key in f['base_data'].keys():
                    data[key] = f['base_data'][key][:]
                for key in f['base_data'].attrs.keys():
                    data[key] = f['base_data'].attrs[key]

            # Create result instance
            result = cls(
                method=method,
                data=data,
                metadata=metadata,
                metrics=metrics,
                timestamp=timestamp,
                content_hash=content_hash
            )

            # Load specialized data
            if 'voltages' in f:
                result.voltages = f['voltages'][:]

            if 'currents' in f:
                result.currents = {name: f['currents'][name][:] for name in f['currents'].keys()}

            if 'conductances' in f:
                result.conductances = {name: f['conductances'][name][:] for name in f['conductances'].keys()}

            if 'synaptic_weights' in f:
                result.synaptic_weights = f['synaptic_weights'][:]

            if 'plasticity_traces' in f:
                result.plasticity_traces = {name: f['plasticity_traces'][name][:]
                                           for name in f['plasticity_traces'].keys()}

            if 'atp_levels' in f:
                result.atp_levels = f['atp_levels'][:]

            if 'glucose_levels' in f:
                result.glucose_levels = f['glucose_levels'][:]

            if 'oxygen_consumption' in f:
                result.oxygen_consumption = f['oxygen_consumption'][:]

            if 'spike_times' in f:
                result.spike_times = [f['spike_times'][name][:] for name in f['spike_times'].keys()]

            if 'isi_distribution' in f:
                result.isi_distribution = f['isi_distribution'][:]

            if 'compartment_voltages' in f:
                result.compartment_voltages = {name: f['compartment_voltages'][name][:]
                                              for name in f['compartment_voltages'].keys()}

            result.spike_rate = f.attrs.get('spike_rate', 0.0)

            if 'compartment_names' in f.attrs:
                result.compartment_names = f.attrs['compartment_names'].split(',')

            return result


@dataclass
class InterventionResult(MechIntResult):
    """
    Results from neural interventions (optogenetics, pharmacology, stimulation).

    Stores intervention parameters, neural responses, and effect metrics.
    """

    # Intervention parameters
    intervention_type: str = 'optogenetics'  # 'optogenetics' | 'pharmacology' | 'stimulation'
    parameters: Dict[str, Any] = field(default_factory=dict)  # Type-specific parameters

    # Temporal profiles
    time: Optional[np.ndarray] = None  # Time vector
    stimulus_profile: Optional[np.ndarray] = None  # Stimulus intensity/concentration over time
    neural_response: Optional[np.ndarray] = None  # Neural response over time

    # Dose-response (for pharmacology)
    doses: Optional[np.ndarray] = None  # Drug concentrations
    responses: Optional[np.ndarray] = None  # Responses per dose
    EC50: Optional[float] = None  # Half-maximal effective concentration
    hill_coefficient: Optional[float] = None  # Hill coefficient

    # Spatial effects (for stimulation)
    spatial_coordinates: Optional[np.ndarray] = None  # (x, y, z) coordinates
    field_strength: Optional[np.ndarray] = None  # Field strength at each point
    activation_volume: Optional[float] = None  # Volume of activated tissue (mm³)

    # Effect metrics
    onset_latency: Optional[float] = None  # Time to response onset (ms)
    peak_response: Optional[float] = None  # Peak response magnitude
    response_duration: Optional[float] = None  # Duration of response (ms)
    effect_size: Optional[float] = None  # Cohen's d or similar

    def save(self, filepath: str):
        """Save intervention result."""
        with h5py.File(filepath, 'w') as f:
            super()._save_base_data(f)

            # Parameters
            f.attrs['intervention_type'] = self.intervention_type
            for key, value in self.parameters.items():
                if isinstance(value, (int, float, str)):
                    f.attrs[f'param_{key}'] = value

            # Temporal data
            if self.time is not None:
                f.create_dataset('time', data=self.time)

            if self.stimulus_profile is not None:
                f.create_dataset('stimulus_profile', data=self.stimulus_profile, compression='gzip')

            if self.neural_response is not None:
                f.create_dataset('neural_response', data=self.neural_response, compression='gzip')

            # Dose-response
            if self.doses is not None:
                f.create_dataset('doses', data=self.doses)

            if self.responses is not None:
                f.create_dataset('responses', data=self.responses)

            # Spatial
            if self.spatial_coordinates is not None:
                f.create_dataset('spatial_coordinates', data=self.spatial_coordinates, compression='gzip')

            if self.field_strength is not None:
                f.create_dataset('field_strength', data=self.field_strength, compression='gzip')

            # Metrics
            f.attrs['EC50'] = self.EC50 if self.EC50 else 0.0
            f.attrs['hill_coefficient'] = self.hill_coefficient if self.hill_coefficient else 0.0
            f.attrs['activation_volume'] = self.activation_volume if self.activation_volume else 0.0
            f.attrs['onset_latency'] = self.onset_latency if self.onset_latency else 0.0
            f.attrs['peak_response'] = self.peak_response if self.peak_response else 0.0
            f.attrs['response_duration'] = self.response_duration if self.response_duration else 0.0
            f.attrs['effect_size'] = self.effect_size if self.effect_size else 0.0

    @classmethod
    def load(cls, filepath: str) -> 'InterventionResult':
        """Load intervention result."""
        with h5py.File(filepath, 'r') as f:
            result = cls(**cls._load_base_data(f))

            result.intervention_type = f.attrs.get('intervention_type', 'optogenetics')

            # Load parameters
            result.parameters = {}
            for key in f.attrs.keys():
                if key.startswith('param_'):
                    result.parameters[key[6:]] = f.attrs[key]

            # Temporal data
            if 'time' in f:
                result.time = f['time'][:]
            if 'stimulus_profile' in f:
                result.stimulus_profile = f['stimulus_profile'][:]
            if 'neural_response' in f:
                result.neural_response = f['neural_response'][:]

            # Dose-response
            if 'doses' in f:
                result.doses = f['doses'][:]
            if 'responses' in f:
                result.responses = f['responses'][:]

            # Spatial
            if 'spatial_coordinates' in f:
                result.spatial_coordinates = f['spatial_coordinates'][:]
            if 'field_strength' in f:
                result.field_strength = f['field_strength'][:]

            # Metrics
            result.EC50 = f.attrs.get('EC50', None)
            result.hill_coefficient = f.attrs.get('hill_coefficient', None)
            result.activation_volume = f.attrs.get('activation_volume', None)
            result.onset_latency = f.attrs.get('onset_latency', None)
            result.peak_response = f.attrs.get('peak_response', None)
            result.response_duration = f.attrs.get('response_duration', None)
            result.effect_size = f.attrs.get('effect_size', None)

            return result


@dataclass
class CriticalityResult(MechIntResult):
    """
    Results from criticality and avalanche analysis.

    Stores avalanche statistics, branching parameters, and power law fits.
    """

    # Avalanche data
    avalanche_sizes: Optional[np.ndarray] = None  # Size of each avalanche
    avalanche_durations: Optional[np.ndarray] = None  # Duration of each avalanche
    avalanche_shapes: Optional[List[np.ndarray]] = None  # Temporal shape of each avalanche

    # Power law analysis
    size_exponent: Optional[float] = None  # Power law exponent for sizes (τ)
    duration_exponent: Optional[float] = None  # Power law exponent for durations
    size_duration_relation: Optional[Tuple[float, float]] = None  # (slope, R²) of size vs duration

    # Branching process
    branching_parameter: Optional[float] = None  # σ (should be ~1 at criticality)
    branching_std: Optional[float] = None  # Standard deviation of branching

    # Criticality metrics
    distance_from_criticality: Optional[float] = None  # |σ - 1|
    kappa: Optional[float] = None  # Shape collapse parameter
    DCC: Optional[float] = None  # Deviation from criticality (DCC)

    # Activity patterns
    activity_matrix: Optional[np.ndarray] = None  # Binary activity (time x neurons)
    avalanche_onsets: Optional[np.ndarray] = None  # Onset times of avalanches

    # Temporal statistics
    inter_avalanche_intervals: Optional[np.ndarray] = None  # Time between avalanches
    mean_avalanche_rate: Optional[float] = None  # Avalanches per unit time

    def save(self, filepath: str):
        """Save criticality result."""
        with h5py.File(filepath, 'w') as f:
            super()._save_base_data(f)

            # Avalanche data
            if self.avalanche_sizes is not None:
                f.create_dataset('avalanche_sizes', data=self.avalanche_sizes)

            if self.avalanche_durations is not None:
                f.create_dataset('avalanche_durations', data=self.avalanche_durations)

            if self.avalanche_shapes:
                shapes_group = f.create_group('avalanche_shapes')
                for i, shape in enumerate(self.avalanche_shapes):
                    shapes_group.create_dataset(f'avalanche_{i}', data=shape)

            # Activity
            if self.activity_matrix is not None:
                f.create_dataset('activity_matrix', data=self.activity_matrix, compression='gzip')

            if self.avalanche_onsets is not None:
                f.create_dataset('avalanche_onsets', data=self.avalanche_onsets)

            if self.inter_avalanche_intervals is not None:
                f.create_dataset('inter_avalanche_intervals', data=self.inter_avalanche_intervals)

            # Metrics
            f.attrs['size_exponent'] = self.size_exponent if self.size_exponent else 0.0
            f.attrs['duration_exponent'] = self.duration_exponent if self.duration_exponent else 0.0
            f.attrs['branching_parameter'] = self.branching_parameter if self.branching_parameter else 0.0
            f.attrs['branching_std'] = self.branching_std if self.branching_std else 0.0
            f.attrs['distance_from_criticality'] = (self.distance_from_criticality
                                                    if self.distance_from_criticality else 0.0)
            f.attrs['kappa'] = self.kappa if self.kappa else 0.0
            f.attrs['DCC'] = self.DCC if self.DCC else 0.0
            f.attrs['mean_avalanche_rate'] = self.mean_avalanche_rate if self.mean_avalanche_rate else 0.0

            if self.size_duration_relation:
                f.attrs['size_duration_slope'] = self.size_duration_relation[0]
                f.attrs['size_duration_r2'] = self.size_duration_relation[1]

    @classmethod
    def load(cls, filepath: str) -> 'CriticalityResult':
        """Load criticality result."""
        with h5py.File(filepath, 'r') as f:
            result = cls(**cls._load_base_data(f))

            # Avalanche data
            if 'avalanche_sizes' in f:
                result.avalanche_sizes = f['avalanche_sizes'][:]
            if 'avalanche_durations' in f:
                result.avalanche_durations = f['avalanche_durations'][:]
            if 'avalanche_shapes' in f:
                result.avalanche_shapes = [f['avalanche_shapes'][name][:]
                                          for name in f['avalanche_shapes'].keys()]

            # Activity
            if 'activity_matrix' in f:
                result.activity_matrix = f['activity_matrix'][:]
            if 'avalanche_onsets' in f:
                result.avalanche_onsets = f['avalanche_onsets'][:]
            if 'inter_avalanche_intervals' in f:
                result.inter_avalanche_intervals = f['inter_avalanche_intervals'][:]

            # Metrics
            result.size_exponent = f.attrs.get('size_exponent', None)
            result.duration_exponent = f.attrs.get('duration_exponent', None)
            result.branching_parameter = f.attrs.get('branching_parameter', None)
            result.branching_std = f.attrs.get('branching_std', None)
            result.distance_from_criticality = f.attrs.get('distance_from_criticality', None)
            result.kappa = f.attrs.get('kappa', None)
            result.DCC = f.attrs.get('DCC', None)
            result.mean_avalanche_rate = f.attrs.get('mean_avalanche_rate', None)

            if 'size_duration_slope' in f.attrs and 'size_duration_r2' in f.attrs:
                result.size_duration_relation = (
                    f.attrs['size_duration_slope'],
                    f.attrs['size_duration_r2']
                )

            return result


@dataclass
class MultifractalResult(MechIntResult):
    """
    Results from multifractal analysis (MF-DFA, WTMM).

    Stores singularity spectra, scaling exponents, and multifractal metrics.
    """

    # Analysis type
    analysis_method: str = 'mfdfa'  # 'mfdfa' | 'wtmm' | 'wavelet'

    # Singularity spectrum
    alpha_values: Optional[np.ndarray] = None  # Hölder exponents
    f_alpha_values: Optional[np.ndarray] = None  # Fractal dimensions f(α)

    # Scaling exponents
    q_values: Optional[np.ndarray] = None  # Moment orders
    tau_q: Optional[np.ndarray] = None  # Scaling exponents τ(q)
    h_q: Optional[np.ndarray] = None  # Generalized Hurst exponents h(q)

    # MF-DFA specific
    scales: Optional[np.ndarray] = None  # Scale values
    fluctuation_functions: Optional[np.ndarray] = None  # F_q(s) for each q

    # WTMM specific
    partition_functions: Optional[np.ndarray] = None  # Z_q(a) for each q, scale a
    wtmm_skeleton: Optional[Dict[str, np.ndarray]] = None  # Maxima lines

    # Multifractal metrics
    multifractal_width: Optional[float] = None  # Δα = α_max - α_min
    asymmetry: Optional[float] = None  # (α_max - α_0) - (α_0 - α_min)
    hurst_exponent: Optional[float] = None  # h(q=2) or H
    intermittency: Optional[float] = None  # Degree of intermittency

    # Comparison to monofractal
    is_multifractal: Optional[bool] = None  # Statistical test result
    multifractality_test_p: Optional[float] = None  # p-value of test

    def save(self, filepath: str):
        """Save multifractal result."""
        with h5py.File(filepath, 'w') as f:
            super()._save_base_data(f)

            f.attrs['analysis_method'] = self.analysis_method

            # Singularity spectrum
            if self.alpha_values is not None:
                f.create_dataset('alpha_values', data=self.alpha_values)
            if self.f_alpha_values is not None:
                f.create_dataset('f_alpha_values', data=self.f_alpha_values)

            # Scaling exponents
            if self.q_values is not None:
                f.create_dataset('q_values', data=self.q_values)
            if self.tau_q is not None:
                f.create_dataset('tau_q', data=self.tau_q)
            if self.h_q is not None:
                f.create_dataset('h_q', data=self.h_q)

            # MF-DFA
            if self.scales is not None:
                f.create_dataset('scales', data=self.scales)
            if self.fluctuation_functions is not None:
                f.create_dataset('fluctuation_functions', data=self.fluctuation_functions,
                               compression='gzip')

            # WTMM
            if self.partition_functions is not None:
                f.create_dataset('partition_functions', data=self.partition_functions,
                               compression='gzip')

            if self.wtmm_skeleton:
                wtmm_group = f.create_group('wtmm_skeleton')
                for key, value in self.wtmm_skeleton.items():
                    wtmm_group.create_dataset(key, data=value, compression='gzip')

            # Metrics
            f.attrs['multifractal_width'] = self.multifractal_width if self.multifractal_width else 0.0
            f.attrs['asymmetry'] = self.asymmetry if self.asymmetry else 0.0
            f.attrs['hurst_exponent'] = self.hurst_exponent if self.hurst_exponent else 0.5
            f.attrs['intermittency'] = self.intermittency if self.intermittency else 0.0
            f.attrs['is_multifractal'] = self.is_multifractal if self.is_multifractal else False
            f.attrs['multifractality_test_p'] = (self.multifractality_test_p
                                                 if self.multifractality_test_p else 1.0)

    @classmethod
    def load(cls, filepath: str) -> 'MultifractalResult':
        """Load multifractal result."""
        with h5py.File(filepath, 'r') as f:
            result = cls(**cls._load_base_data(f))

            result.analysis_method = f.attrs.get('analysis_method', 'mfdfa')

            # Singularity spectrum
            if 'alpha_values' in f:
                result.alpha_values = f['alpha_values'][:]
            if 'f_alpha_values' in f:
                result.f_alpha_values = f['f_alpha_values'][:]

            # Scaling exponents
            if 'q_values' in f:
                result.q_values = f['q_values'][:]
            if 'tau_q' in f:
                result.tau_q = f['tau_q'][:]
            if 'h_q' in f:
                result.h_q = f['h_q'][:]

            # MF-DFA
            if 'scales' in f:
                result.scales = f['scales'][:]
            if 'fluctuation_functions' in f:
                result.fluctuation_functions = f['fluctuation_functions'][:]

            # WTMM
            if 'partition_functions' in f:
                result.partition_functions = f['partition_functions'][:]

            if 'wtmm_skeleton' in f:
                result.wtmm_skeleton = {name: f['wtmm_skeleton'][name][:]
                                       for name in f['wtmm_skeleton'].keys()}

            # Metrics
            result.multifractal_width = f.attrs.get('multifractal_width', None)
            result.asymmetry = f.attrs.get('asymmetry', None)
            result.hurst_exponent = f.attrs.get('hurst_exponent', None)
            result.intermittency = f.attrs.get('intermittency', None)
            result.is_multifractal = f.attrs.get('is_multifractal', None)
            result.multifractality_test_p = f.attrs.get('multifractality_test_p', None)

            return result


__all__ = [
    'BiophysicalResult',
    'InterventionResult',
    'CriticalityResult',
    'MultifractalResult',
]
