from __future__ import annotations

import base64
from pathlib import Path

import pytest

from neuros.cloud import LocalStorage


def test_run_ids_cannot_escape_storage_root(tmp_path: Path) -> None:
    storage = LocalStorage(tmp_path / "runs")
    for run_id in ("../escape", "..", "/absolute", "nested/run", "nested\\run"):
        with pytest.raises(ValueError, match="run_id"):
            storage.upload_metrics({"accuracy": 0.5}, run_id)
    assert not (tmp_path / "escape").exists()


def test_plaintext_mode_is_explicit_and_base64_is_only_encoding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("NEUROS_ENCRYPTION_KEY", raising=False)
    storage = LocalStorage(tmp_path)
    storage.upload_metrics({"accuracy": 0.75}, "run-1")
    storage.stream_results([(1.0, 2, 0.8, 0.01)], "run-1")

    run_dir = tmp_path / "run-1"
    assert (run_dir / "metrics.json").exists()
    encoded = (run_dir / "metrics.b64").read_text(encoding="utf-8")
    assert b'"accuracy": 0.75' in base64.b64decode(encoded)
    assert (run_dir / "stream.log").exists()
    assert (run_dir / "stream.b64").exists()
    assert not (run_dir / "metrics.enc").exists()
    assert not (run_dir / "stream.enc").exists()


def test_encrypted_mode_writes_no_plaintext_sidecars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fernet_module = pytest.importorskip("cryptography.fernet")
    key = fernet_module.Fernet.generate_key()
    monkeypatch.setenv("NEUROS_ENCRYPTION_KEY", key.decode("ascii"))

    storage = LocalStorage(tmp_path)
    storage.upload_metrics({"accuracy": 0.75}, "secure")
    storage.stream_results([(1.0, 2, 0.8, 0.01)], "secure")

    run_dir = tmp_path / "secure"
    assert not (run_dir / "metrics.json").exists()
    assert not (run_dir / "metrics.b64").exists()
    assert not (run_dir / "stream.log").exists()
    assert not (run_dir / "stream.b64").exists()

    encrypted_metrics = (run_dir / "metrics.enc").read_text(encoding="utf-8")
    decrypted_metrics = fernet_module.Fernet(key).decrypt(encrypted_metrics.encode("ascii"))
    assert b'"accuracy": 0.75' in decrypted_metrics

    tokens = (run_dir / "stream.enc").read_text(encoding="utf-8").splitlines()
    assert len(tokens) == 1
    decrypted_stream = fernet_module.Fernet(key).decrypt(tokens[0].encode("ascii"))
    assert decrypted_stream == b"1.000\t2\t0.800\t0.010000\n"


def test_switching_to_encrypted_mode_removes_existing_plaintext(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fernet_module = pytest.importorskip("cryptography.fernet")
    storage = LocalStorage(tmp_path)
    monkeypatch.delenv("NEUROS_ENCRYPTION_KEY", raising=False)
    storage.upload_metrics({"value": 1.0}, "run")
    storage.stream_results([(1.0, 1, 1.0, 0.0)], "run")

    monkeypatch.setenv(
        "NEUROS_ENCRYPTION_KEY", fernet_module.Fernet.generate_key().decode("ascii")
    )
    storage.upload_metrics({"value": 2.0}, "run")
    storage.stream_results([(2.0, 2, 1.0, 0.0)], "run")

    run_dir = tmp_path / "run"
    assert not (run_dir / "metrics.json").exists()
    assert not (run_dir / "metrics.b64").exists()
    assert not (run_dir / "stream.log").exists()
    assert not (run_dir / "stream.b64").exists()


def test_invalid_encryption_key_fails_before_writing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEUROS_ENCRYPTION_KEY", "not-a-fernet-key")
    storage = LocalStorage(tmp_path)
    with pytest.raises(ValueError, match="Fernet key"):
        storage.upload_metrics({"value": 1.0}, "secure")
    assert not (tmp_path / "secure").exists()


def test_database_backup_fails_loudly_and_copies_valid_files(tmp_path: Path) -> None:
    storage = LocalStorage(tmp_path / "runs")
    with pytest.raises(FileNotFoundError):
        storage.upload_database(tmp_path / "missing.sqlite")

    source = tmp_path / "source.sqlite"
    source.write_bytes(b"sqlite-test")
    storage.upload_database(source)
    assert (tmp_path / "runs" / "source.sqlite").read_bytes() == b"sqlite-test"
