from pathlib import Path

import pytest

from neuros.cli.recording_commands import inspect_archive, record_config, replay_archive
from neuros.recording import SessionArchiveReader


CONFIG = Path("configs/examples/mock_bci.yaml")


@pytest.mark.asyncio
async def test_record_inspect_replay_workflow(tmp_path: Path):
    archive = tmp_path / "session"
    recorded = await record_config(
        CONFIG,
        archive,
        session_id="ci-session",
        duration_s=0.05,
    )
    assert recorded["status"] == "complete"
    assert set(recorded["streams"]) == {"eeg"}
    assert recorded["streams"]["eeg"] > 0

    reader = SessionArchiveReader(archive)
    descriptor = reader.descriptor("eeg")
    assert descriptor.stream_id == "eeg"
    assert descriptor.metadata["source_stream_id"] == "mockdriver"
    first_frame = next(iter(reader.iter_frames("eeg")))
    assert first_frame.stream_id == "eeg"
    assert first_frame.metadata["source_stream_id"] == "mockdriver"

    inspected = inspect_archive(archive, verify_hashes=True)
    assert inspected["integrity"] == "verified"
    assert inspected["streams"]["eeg"] == recorded["streams"]["eeg"]

    replayed = await replay_archive(archive, CONFIG)
    assert replayed["state"] == "stopped"
    assert replayed["nodes"]["decoder:primary"]["processed"] > 0
    assert replayed["edges"]["source:eeg->transform:eeg:0"]["dropped"] == 0
