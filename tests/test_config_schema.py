import pytest

from neuros.config import PipelineConfig
from neuros.errors import ConfigurationError
from neuros.runtime import OverflowPolicy


def test_pipeline_config_parses_named_streams_and_runtime_policy():
    config = PipelineConfig.from_mapping(
        {
            "schema_version": 1,
            "streams": [
                {
                    "id": "eeg",
                    "source": {"plugin": "mock", "options": {"sample_rate": 250}},
                    "transforms": [{"plugin": "bandpass"}],
                }
            ],
            "decoder": {"plugin": "simple"},
            "runtime": {"queue_capacity": 8, "overflow_policy": "drop_oldest"},
        }
    )
    assert config.streams[0].stream_id == "eeg"
    assert config.runtime.queue_capacity == 8
    assert config.runtime.overflow_policy is OverflowPolicy.DROP_OLDEST


def test_pipeline_config_rejects_duplicate_stream_ids():
    raw_stream = {"id": "eeg", "source": {"plugin": "mock"}}
    with pytest.raises(ConfigurationError):
        PipelineConfig.from_mapping(
            {
                "schema_version": 1,
                "streams": [raw_stream, raw_stream],
                "decoder": {"plugin": "simple"},
            }
        )
