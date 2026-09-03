"""Exact-head native aligned-execution qualification fixture.

Run only after installing the wheel built from the same GitHub Actions checkout.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import struct
import sys
import tempfile
from pathlib import Path

import neuros_runtime_native as native


def encode_f32(values: list[float]) -> bytes:
    return b"".join(struct.pack("<f", value) for value in values)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_public_dataset_module():
    module_path = Path("packages/neuros/src/neuros/dataset.py").resolve()
    spec = importlib.util.spec_from_file_location(
        "neuros_runtime_aligned_execution_smoke", module_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def assert_window_values(window, expected: list[float]) -> None:
    actual = window.to_pyarrow().to_pylist()
    assert actual == expected, (actual, expected)


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        fmri_values = [1000.0 + value for value in range(40)]
        behavior_values = [2000.0 + value for value in range(40)]
        fmri_bytes = encode_f32(fmri_values)
        behavior_bytes = encode_f32(behavior_values)
        (root / "fmri.f32").write_bytes(fmri_bytes)
        (root / "behavior.f32").write_bytes(behavior_bytes)
        fmri_sha256 = sha256_bytes(fmri_bytes)
        behavior_sha256 = sha256_bytes(behavior_bytes)

        manifest = {
            "schema_version": 1,
            "dataset_id": "aligned-execution-ci",
            "records": [
                {
                    "id": "fmri-run-01",
                    "subject": "sub-01",
                    "modality": "fmri",
                    "sync_group": "sub-01/run-01",
                    "path": "fmri.f32",
                    "source_sha256": fmri_sha256,
                    "dtype": "float32-le",
                    "shape": [10, 4],
                    "sampling_hz": 0.5,
                    "clock": {
                        "id": "scanner-clock",
                        "start_ns": 0,
                        "period_ns": 2_000_000_000,
                    },
                },
                {
                    "id": "behavior-run-01",
                    "subject": "sub-01",
                    "modality": "behavior",
                    "sync_group": "sub-01/run-01",
                    "path": "behavior.f32",
                    "source_sha256": behavior_sha256,
                    "dtype": "float32-le",
                    "shape": [40, 1],
                    "sampling_hz": 2.0,
                    "clock": {
                        "id": "behavior-clock",
                        "start_ns": 0,
                        "period_ns": 500_000_000,
                    },
                },
            ],
        }
        (root / "neuros.dataset.json").write_text(json.dumps(manifest))

        dataset = native.NativeDataset.open(root)
        plan = dataset.plan_aligned(
            sync_group="sub-01/run-01",
            modalities=["fmri", "behavior"],
            duration_ns=4_000_000_000,
            stride_ns=2_000_000_000,
        )
        plan_payload = json.loads(plan.to_json())
        plan_sha256 = plan.sha256
        assert plan.window_count == 9
        assert len(plan_sha256) == 64

        entries = {entry["modality"]: entry for entry in plan_payload["entries"]}
        stream = dataset.stream_aligned(plan=plan, prefetch=2)
        emitted = 0
        for index, batch in enumerate(stream):
            emitted += 1
            assert batch.window_index == index
            assert batch.plan_sha256 == plan_sha256
            assert batch.dataset_content_sha256 == plan.dataset_content_sha256
            assert batch.manifest_sha256 == plan.manifest_sha256
            assert batch.sync_group == "sub-01/run-01"
            assert batch.start_ns == index * plan.stride_ns
            assert batch.end_ns == batch.start_ns + plan.duration_ns
            assert batch.modalities == ["behavior", "fmri"]

            behavior = batch.window("behavior")
            fmri = batch.window("fmri")
            assert behavior is not None and fmri is not None
            assert batch.window("missing") is None

            for modality, window in (("behavior", behavior), ("fmri", fmri)):
                entry = entries[modality]
                expected_start = entry["start_frame"] + index * entry["frame_stride"]
                expected_stop = expected_start + entry["frames_per_window"]
                assert window.start_frame == expected_start
                assert window.end_frame_exclusive == expected_stop
                assert window.verified_source_sha256 == entry["source_sha256"]
                assert (
                    window.verified_dataset_content_sha256
                    == plan.dataset_content_sha256
                )

            fmri_start = fmri.start_frame * 4
            fmri_stop = fmri.end_frame_exclusive * 4
            assert_window_values(fmri, fmri_values[fmri_start:fmri_stop])
            assert_window_values(
                behavior,
                behavior_values[
                    behavior.start_frame : behavior.end_frame_exclusive
                ],
            )
        assert emitted == plan.window_count

        # Public SDK parity must consume the exact native plan object rather than
        # serialize/replan it behind the user's back.
        dataset_module = load_public_dataset_module()
        study = dataset_module.Dataset.open(root)
        public_plan = study.plan_aligned(
            sync_group="sub-01/run-01",
            modalities=["fmri", "behavior"],
            duration_ns=4_000_000_000,
            stride_ns=2_000_000_000,
        )
        first_public = next(study.stream_aligned(public_plan, prefetch=1))
        assert first_public.plan_sha256 == public_plan.sha256
        assert first_public.dataset_content_sha256 == public_plan.dataset_content_sha256
        assert first_public.manifest_sha256 == public_plan.manifest_sha256
        assert first_public.modalities == ("behavior", "fmri")
        assert_window_values(
            first_public.window("fmri")._native_window,
            fmri_values[:8],
        )
        assert_window_values(
            first_public.window("behavior")._native_window,
            behavior_values[:8],
        )
        assert len(first_public.fmri) == 8
        assert len(first_public.behavior) == 8
        provenance = first_public.provenance
        assert provenance["plan_sha256"] == public_plan.sha256
        assert set(provenance["modalities"]) == {"behavior", "fmri"}

        try:
            dataset.stream_aligned(plan=plan, prefetch=0)
        except RuntimeError as error:
            assert "prefetch" in str(error)
        else:  # pragma: no cover - fail closed
            raise AssertionError("aligned prefetch=0 unexpectedly succeeded")

        # Keep a previously verified mmap alive, mutate the physical file, and
        # prove aligned execution does not trust the cached verification state.
        cached_stream = dataset.stream(
            modalities=["fmri"], window=2, stride=2, prefetch=1
        )
        cached_window = next(cached_stream)
        assert cached_window.verified_source_sha256 == fmri_sha256
        mutated = bytearray(fmri_bytes)
        mutated[0] ^= 0xFF
        (root / "fmri.f32").write_bytes(mutated)
        try:
            dataset.stream_aligned(plan=plan, prefetch=1)
        except RuntimeError as error:
            assert "SHA-256" in str(error) or "hash" in str(error).lower()
        else:  # pragma: no cover - fail closed
            raise AssertionError("mutation after cached verification was not rejected")
        assert cached_window.record_id == "fmri-run-01"

        # Dropping a bounded stream must not require draining it. The Rust unit
        # contract verifies worker cancellation; here we exercise Python ownership.
        del cached_stream
        gc.collect()


if __name__ == "__main__":
    main()
