"""
Drivers provide a common interface for acquiring neural signals.  A driver
abstracts away hardware details and exposes an asynchronous iterator
yielding samples as NumPy arrays.
"""

from neuros.drivers.base_driver import BaseDriver  # noqa: F401
from neuros.drivers.mock_driver import MockDriver  # noqa: F401
from neuros.drivers.brainflow_driver import BrainFlowDriver  # noqa: F401

# additional drivers
from neuros.drivers.video_driver import VideoDriver  # noqa: F401
from neuros.drivers.dataset_driver import DatasetDriver  # noqa: F401
from neuros.drivers.motion_sensor_driver import MotionSensorDriver  # noqa: F401

# biosignal drivers
from neuros.drivers.ecog_driver import ECoGDriver  # noqa: F401
from neuros.drivers.emg_driver import EMGDriver  # noqa: F401
from neuros.drivers.eog_driver import EOGDriver  # noqa: F401
from neuros.drivers.calcium_imaging_driver import CalciumImagingDriver  # noqa: F401

# newly added biosignal and audio drivers
from neuros.drivers.ecg_driver import ECGDriver  # noqa: F401
from neuros.drivers.gsr_driver import GSRDriver  # noqa: F401
from neuros.drivers.respiration_driver import RespirationDriver  # noqa: F401
from neuros.drivers.hormone_driver import HormoneDriver  # noqa: F401
from neuros.drivers.audio_driver import AudioDriver  # noqa: F401

# behavioural and optical drivers
from neuros.drivers.phone_driver import PhoneDriver  # noqa: F401
from neuros.drivers.fnirs_driver import FnirsDriver  # noqa: F401
