from __future__ import annotations

import numpy as np
import pytest

from neuros.drivers.unicorn_api_sim import UnicornApiSimError, UnicornPythonApiSimulator


def test_api_simulator_requires_explicit_synthetic_serial_identity():
    with pytest.raises(ValueError):
        UnicornPythonApiSimulator(serial="REAL-LOOKING")
    api = UnicornPythonApiSimulator()
    assert api.get_available_devices() == ["SIM-UNICORN-0001"]
    assert api.get_device_information().number_of_eeg_channels == 8


def test_configuration_and_channel_lookup_follow_enabled_scan_order():
    api = UnicornPythonApiSimulator()
    config = api.get_configuration()
    config = config.with_channel("Gyroscope Z", enabled=False)
    api.set_configuration(config)
    assert api.get_number_of_acquired_channels() == 16
    assert api.get_channel_index("EEG 1") == 0
    assert api.get_channel_index("Battery Level") == 14
    with pytest.raises(KeyError):
        api.get_channel_index("Gyroscope Z")


def test_acquisition_lifecycle_and_get_data_shape():
    api = UnicornPythonApiSimulator(seed=13)
    with pytest.raises(UnicornApiSimError) as exc:
        api.get_data(5)
    assert exc.value.code == "operation_not_allowed"
    api.start_acquisition(False)
    data = api.get_data(7)
    assert data.shape == (7, 17)
    assert data.dtype == np.float32
    with pytest.raises(UnicornApiSimError):
        api.set_configuration(api.get_configuration())
    api.stop_acquisition()


def test_test_signal_mode_is_rectangular_and_explicitly_simulator_defined():
    api = UnicornPythonApiSimulator(
        seed=17,
        test_signal_frequency_hz=5.0,
        test_signal_amplitude_uv=80.0,
    )
    api.start_acquisition(True)
    data = api.get_data(100)
    eeg = data[:, :8]
    assert set(np.unique(eeg)).issubset({-80.0, 80.0})
    assert np.allclose(eeg[:, 0], eeg[0, 0])


def test_digital_outputs_are_8_bit_state():
    api = UnicornPythonApiSimulator()
    api.set_digital_outputs(170)
    assert api.get_digital_outputs() == 170
    with pytest.raises(ValueError):
        api.set_digital_outputs(256)


def test_injected_api_errors_are_one_shot_and_recovery_is_explicit():
    api = UnicornPythonApiSimulator(seed=19)
    api.start_acquisition(False)
    for code in ("buffer_underflow", "buffer_overflow", "connection_problem"):
        api.inject_next_error(code)
        with pytest.raises(UnicornApiSimError) as exc:
            api.get_data(5)
        assert exc.value.code == code
        recovered = api.get_data(5)
        assert recovered.shape == (5, 17)


def test_binary_scan_buffer_matches_enabled_channel_count():
    api = UnicornPythonApiSimulator(seed=23)
    config = api.get_configuration().with_channel("Accelerometer Z", enabled=False)
    api.set_configuration(config)
    api.start_acquisition(False)
    payload = api.get_data_bytes(4)
    assert len(payload) == 4 * 16 * 4
