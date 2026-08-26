"""
Drivers provide a common interface for acquiring neural signals. A driver
abstracts away hardware details and exposes asynchronous neural streams.
"""

from neuros.drivers.base_driver import BaseDriver  # noqa: F401
from neuros.drivers.brainflow_driver import BrainFlowDriver  # noqa: F401
from neuros.drivers.lsl_driver import LSLDriver  # noqa: F401
from neuros.drivers.mock_driver import MockDriver  # noqa: F401
from neuros.drivers.synthetic_eeg import (  # noqa: F401
    ArtifactKind,
    SyntheticEEGBlock,
    SyntheticEEGConfig,
    SyntheticEEGGenerator,
)
from neuros.drivers.synthetic_eeg_driver import SyntheticEEGDriver  # noqa: F401
from neuros.drivers.unicorn_api_sim import (  # noqa: F401
    AmplifierChannelSim,
    AmplifierConfigurationSim,
    DeviceInformationSim,
    UnicornApiSimError,
    UnicornPythonApiSimulator,
)
from neuros.drivers.unicorn_compatibility import (  # noqa: F401
    UnicornCompatibilityReport,
    UnicornCompatibilitySurface,
    run_unicorn_compatibility_suite,
)
from neuros.drivers.unicorn_hybrid_black_sim import (  # noqa: F401
    UNICORN_ACCEL_NAMES,
    UNICORN_AUX_NAMES,
    UNICORN_DEVICE17_NAMES,
    UNICORN_EEG_API_NAMES,
    UNICORN_GYRO_NAMES,
    UNICORN_RECORDER19_NAMES,
    UNICORN_SCALP_LABELS,
    UnicornConformanceReport,
    UnicornHybridBlackBlock,
    UnicornHybridBlackSimulationConfig,
    UnicornHybridBlackSimulator,
    UnicornHybridBlackSpec,
    validate_unicorn_block,
)
from neuros.drivers.unicorn_network_sim import (  # noqa: F401
    API_FROM_RAW_UDP_INDICES,
    BANDPOWER_BANDS,
    BANDPOWER_FEATURE_COUNT,
    RAW_UDP_CHANNEL_COUNT,
    RAW_UDP_FROM_API_INDICES,
    RAW_UDP_PAYLOAD_BYTES,
    UNICORN_RAW_UDP_NAMES,
    BandpowerFrame,
    UnicornBandpowerReferenceStream,
    api17_scan_to_raw_udp_order,
    compute_unicorn_bandpower_payload,
    decode_unicorn_bandpower_ascii,
    decode_unicorn_udp_scan,
    encode_unicorn_bandpower_ascii,
    encode_unicorn_udp_scan,
    raw_udp_scan_to_api17_order,
)
from neuros.drivers.unicorn_receiver_guard import (  # noqa: F401
    FLOAT32_EXACT_INTEGER_MAX,
    UnicornRawUdpGuard,
    UnicornRawUdpGuardConfig,
    UnicornRawUdpGuardState,
    UnicornRawUdpObservation,
)
from neuros.drivers.unicorn_trace import (  # noqa: F401
    UnicornRawUdpTraceSummary,
    UnicornTraceContractComparison,
    UnicornTraceDeltaReport,
    analyze_unicorn_raw_udp_trace,
    compare_unicorn_trace_summaries,
    compare_unicorn_trace_to_nominal_contract,
)
from neuros.drivers.unicorn_transport_sim import (  # noqa: F401
    FAULT_PROFILES,
    DeterministicPacketFaultEngine,
    ScheduledDatagram,
    UnicornBandpowerUdpStreamSimulator,
    UnicornRawUdpStreamSimulator,
    UnicornUdpFaultProfile,
    get_unicorn_udp_fault_profile,
)

# additional drivers
from neuros.drivers.dataset_driver import DatasetDriver  # noqa: F401
from neuros.drivers.motion_sensor_driver import MotionSensorDriver  # noqa: F401
from neuros.drivers.video_driver import VideoDriver  # noqa: F401

# biosignal drivers
from neuros.drivers.calcium_imaging_driver import CalciumImagingDriver  # noqa: F401
from neuros.drivers.ecog_driver import ECoGDriver  # noqa: F401
from neuros.drivers.emg_driver import EMGDriver  # noqa: F401
from neuros.drivers.eog_driver import EOGDriver  # noqa: F401

# newly added biosignal and audio drivers
from neuros.drivers.audio_driver import AudioDriver  # noqa: F401
from neuros.drivers.ecg_driver import ECGDriver  # noqa: F401
from neuros.drivers.gsr_driver import GSRDriver  # noqa: F401
from neuros.drivers.hormone_driver import HormoneDriver  # noqa: F401
from neuros.drivers.respiration_driver import RespirationDriver  # noqa: F401

# behavioural and optical drivers
from neuros.drivers.fnirs_driver import FnirsDriver  # noqa: F401
from neuros.drivers.phone_driver import PhoneDriver  # noqa: F401
