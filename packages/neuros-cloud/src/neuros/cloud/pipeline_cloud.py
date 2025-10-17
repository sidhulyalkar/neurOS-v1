"""
Constellation pipeline orchestration utilities.

This module provides high‑level functions to run a multi‑modal
Constellation pipeline that ingests data from multiple simulated or
real devices, synchronises streams, publishes events to Kafka,
writes raw data to cloud‑native formats (NWB and OME‑Zarr), and
kicks off a distributed training job on SageMaker using WebDataset
shards.  Observability is integrated via Prometheus metrics and
synthesised fault injection can be enabled for robustness testing.

Note that this module contains illustrative stubs rather than fully
operational implementations.  Many components depend on external
infrastructure (Kafka brokers, S3 buckets, Iceberg tables and
SageMaker) which must be provisioned and configured separately.  The
functions herein demonstrate how such pieces could be wired together
in a future Constellation deployment.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
from prometheus_client import Counter, Histogram

from ..drivers.audio_driver import AudioDriver
from ..drivers.video_driver import VideoDriver
from ..drivers.gsr_driver import GSRDriver
from ..drivers.mock_driver import MockDriver
from ..drivers.phone_driver import PhoneDriver
from ..drivers.fnirs_driver import FnirsDriver

# Optional drivers may not be present in minimal installations.  Import them
# lazily and fall back to None if unavailable.
try:
    from ..drivers.respiration_driver import RespirationDriver  # type: ignore
except Exception:  # pragma: no cover
    RespirationDriver = None  # type: ignore

try:
    from ..drivers.ecg_driver import ECGDriver  # type: ignore
except Exception:  # pragma: no cover
    ECGDriver = None  # type: ignore

# Import KafkaWriter lazily so that the module can be imported even when
# confluent_kafka is not installed.  If the import fails, set
# KafkaWriter to None and rely on the NoopWriter fallback when Kafka
# is not available or when ``--no-kafka`` is passed.
try:
    from ..ingest.kafka_writer import KafkaWriter  # type: ignore
except Exception:  # pragma: no cover
    KafkaWriter = None  # type: ignore
from ..ingest.noop_writer import NoopWriter
from ..io.nwb_writer import write_nwb_file
from ..io.zarr_writer import write_ome_zarr
from ..export.webdataset_exporter import export_to_webdataset
# Import Petastorm exporter lazily to avoid ImportError when the
# petastorm package is not installed.  Set to None if unavailable.
try:
    from ..export.petastorm_exporter import export_to_petastorm  # type: ignore
except Exception:  # pragma: no cover
    export_to_petastorm = None  # type: ignore

# Import SageMaker launcher lazily.  When the sagemaker package is not
# installed this import may fail.  Provide a stub fallback so that
# ``launch_training`` can still be called without crashing.
try:
    from ..training.sagemaker_launcher import launch_training  # type: ignore
except Exception:  # pragma: no cover
    def launch_training(job_name: str, config_path: Optional[str]) -> None:
        logger.warning(
            "SageMaker launcher not available; skipping training job '%s'",
            job_name,
        )

logger = logging.getLogger(__name__)

### Prometheus metrics ###

INGEST_COUNT = Counter(
    "neuros_constellation_ingest_total",
    "Total number of samples ingested per modality",
    ["modality"],
)

INGEST_LATENCY = Histogram(
    "neuros_constellation_ingest_latency_seconds",
    "Latency of ingestion loop per modality",
    ["modality"],
)

STREAM_LATENCY = Histogram(
    "neuros_constellation_stream_latency_seconds",
    "Time taken to stream a batch of events to Kafka",
)


@dataclass
class SensorCollector:
    """Collects samples from a driver for later storage.

    Attributes
    ----------
    modality : str
        Name of the modality (e.g. 'eeg', 'video').
    timestamps : list[float]
        Sample timestamps.
    data : list[np.ndarray]
        Sample payloads.
    """

    modality: str
    timestamps: List[float] = field(default_factory=list)
    data: List[np.ndarray] = field(default_factory=list)

    def append(self, ts: float, payload: np.ndarray) -> None:
        self.timestamps.append(ts)
        self.data.append(payload)


async def ingest_driver(
    modality: str,
    driver,
    kafka: KafkaWriter,
    topic_prefix: str,
    subject_id: str,
    session_id: str,
    sensor_id: str,
    collector: SensorCollector,
    duration: float,
    fault_injection: bool = False,
) -> None:
    """Ingest data from a driver and stream to Kafka.

    This coroutine reads samples from the given driver for the specified
    duration and publishes a JSON event to Kafka for each sample.  A
    ``SensorCollector`` accumulates the raw data so that it can be
    written to NWB/Zarr after ingestion completes.  Optional fault
    injection simulates dropped packets or clock jumps.

    Parameters
    ----------
    modality : str
        Name of the modality.
    driver : BaseDriver
        An instantiated driver providing asynchronous samples.
    kafka : KafkaWriter
        Kafka producer used to send events.
    topic_prefix : str
        Prefix for Kafka topics (e.g. 'raw'); the full topic will be
        ``f"{topic_prefix}.{modality}.{sensor_id}"``.
    subject_id : str
        Identifier for the subject.
    session_id : str
        Identifier for the recording session.
    sensor_id : str
        Identifier for the sensor instance; used in the topic name.
    collector : SensorCollector
        Object used to store timestamps and payloads.
    duration : float
        Duration in seconds to ingest data.
    fault_injection : bool, optional
        If True, randomly drop samples or perturb timestamps to
        simulate network jitter and device failures.
    """
    topic = f"{topic_prefix}.{modality}.{sensor_id}"
    await driver.start()
    start_time = time.perf_counter()
    seq = 0
    async for timestamp, payload in driver:
        now = time.perf_counter()
        if now - start_time > duration:
            break
        # optional fault injection: drop ~5% of samples
        if fault_injection and np.random.rand() < 0.05:
            continue
        # record raw data
        collector.append(timestamp, payload)
        # build event dict
        event_dict = {
            "subject_id": subject_id,
            "session_id": session_id,
            "device_id": sensor_id,
            "modality": modality,
            "timestamp_ns": int(timestamp * 1e9),
            "seq": seq,
            "payload": payload.tolist(),
        }
        # send to Kafka
        send_start = time.perf_counter()
        kafka.write(topic, event_dict)
        send_end = time.perf_counter()
        STREAM_LATENCY.observe(send_end - send_start)
        # update metrics
        INGEST_COUNT.labels(modality=modality).inc()
        INGEST_LATENCY.labels(modality=modality).observe(send_end - (timestamp))
        seq += 1
    await driver.stop()


async def run_multimodal_ingestion(
    duration: float = 10.0,
    kafka_bootstrap: Optional[str] = "localhost:9092",
    topic_prefix: str = "raw",
    subject_id: str = "subj1",
    session_id: str = "sess1",
    fault_injection: bool = False,
) -> Dict[str, SensorCollector]:
    """Run ingestion for multiple modalities concurrently.

    This function spawns ingestion coroutines for each modality
    (EEG, audio, video, EDA/GSR, fNIRS/HD‑DOT, respiration, ECG, phone)
    and waits until all have completed.  It returns a mapping from
    modality name to the corresponding ``SensorCollector`` containing
    the captured samples.

    Parameters
    ----------
    duration : float, optional
        Number of seconds to ingest data from each driver.  Defaults
        to 10 seconds.
    kafka_bootstrap : str, optional
        Bootstrap servers for the Kafka producer.  Defaults to
        ``localhost:9092``.  If set to ``None`` or an empty string,
        the pipeline runs in a dry‑run mode without Kafka using a
        ``NoopWriter``.
    topic_prefix : str, optional
        Prefix for Kafka topics.  Defaults to "raw".
    subject_id : str, optional
        Subject identifier used in event metadata.  Defaults to
        "subj1".
    session_id : str, optional
        Session identifier used in event metadata.  Defaults to
        "sess1".
    fault_injection : bool, optional
        If True, enable simulated packet loss and clock jitter.

    Returns
    -------
    Dict[str, SensorCollector]
        Mapping from modality to the collector storing the raw data.
    """
    # instantiate drivers with reasonable defaults
    drivers: Dict[str, object] = {
        "eeg": MockDriver(sampling_rate=250.0, channels=8),
        "audio": AudioDriver(sampling_rate=16000.0, frequency=440.0),
        "video": VideoDriver(frame_rate=30.0, resolution=(64, 64), channels=3),
        "eda": GSRDriver(sampling_rate=50.0),
        # fNIRS/HD‑DOT simulated driver
        "fnirs": FnirsDriver(sampling_rate=10.0, channels=16),
        # optional respiration driver may not be available
        "respiration": RespirationDriver(sampling_rate=25.0) if RespirationDriver is not None else None,
        # optional ECG driver
        "ecg": ECGDriver(sampling_rate=250.0) if ECGDriver is not None else None,
        "phone": PhoneDriver(sampling_rate=1.0),
    }
    # remove drivers that may be None (e.g. respiration if module missing)
    drivers = {k: v for k, v in drivers.items() if v is not None}
    collectors: Dict[str, SensorCollector] = {k: SensorCollector(k) for k in drivers}
    # Select a writer.  Use NoopWriter for dry‑run mode or when
    # confluent_kafka is not available.  If kafka_bootstrap is
    # provided but KafkaWriter could not be imported, fall back to
    # NoopWriter and emit a warning.
    if kafka_bootstrap is None or kafka_bootstrap == "":
        kafka: KafkaWriter | NoopWriter = NoopWriter()
        logger.info("No Kafka bootstrap provided; using NoopWriter for ingestion.")
    else:
        if KafkaWriter is None:
            logger.warning(
                "KafkaWriter is unavailable (confluent_kafka not installed); using NoopWriter instead"
            )
            kafka = NoopWriter()
        else:
            kafka = KafkaWriter(bootstrap_servers=kafka_bootstrap)
    tasks = []
    for modality, driver in drivers.items():
        sensor_id = f"{modality}01"
        tasks.append(
            ingest_driver(
                modality,
                driver,
                kafka,
                topic_prefix,
                subject_id,
                session_id,
                sensor_id,
                collectors[modality],
                duration,
                fault_injection,
            )
        )
    await asyncio.gather(*tasks)
    # Close the writer if it defines a close method
    if hasattr(kafka, "close"):
        kafka.close()
    return collectors


def write_raw_data_to_storage(
    collectors: Dict[str, SensorCollector],
    output_dir: str,
    subject_id: str,
    session_id: str,
) -> Dict[str, str]:
    """Write collected raw data to local disk (simulating cloud storage).

    The EEG, EDA, respiration, ECG, fNIRS and phone modalities are
    written into NWB files (one file per modality).  Video frames are
    stacked into an array and written as an OME‑Zarr dataset.  Audio
    samples are written into an NWB file.  This function returns a
    mapping from modality to the path of the created file.

    Parameters
    ----------
    collectors : dict
        Mapping from modality to the SensorCollector with raw data.
    output_dir : str
        Base directory where files are written.  For S3 storage this
        could be a prefix like ``"s3://constellation/raw"``; however
        this function writes to the local filesystem for simplicity.
    subject_id : str
        Subject identifier used in file naming.
    session_id : str
        Session identifier used in file naming.

    Returns
    -------
    Dict[str, str]
        Mapping from modality to the file path of the written data.
    """
    os.makedirs(output_dir, exist_ok=True)
    file_paths: Dict[str, str] = {}
    # handle time series modalities (write NWB)
    time_series_modalities = ["eeg", "eda", "respiration", "ecg", "fnirs", "phone", "audio"]
    for modality in time_series_modalities:
        if modality not in collectors:
            continue
        collector = collectors[modality]
        # compute sampling rate from timestamps (approx)
        if len(collector.timestamps) > 1:
            dt = np.diff(collector.timestamps)
            fs = 1.0 / float(np.mean(dt)) if np.mean(dt) > 0 else 1.0
        else:
            fs = 1.0
        file_path = os.path.join(
            output_dir, f"{subject_id}_{session_id}_{modality}.nwb"
        )
        try:
            # Flatten data: for 1‑D arrays we collect into shape (channels, samples)
            # phone and fnirs produce 1‑D arrays; video is handled separately
            arr_list = []
            for sample in collector.data:
                # sample may be scalar or 1‑D/2‑D
                sample_arr = np.atleast_1d(sample)
                arr_list.append(sample_arr)
            # stack along time axis then transpose to channels × samples
            arr = np.stack(arr_list, axis=0)
            # if single channel, shape = (n_samples, 1) after new axis
            if arr.ndim == 1:
                arr = arr[:, np.newaxis]
            arr = arr.T  # shape (channels, samples)
            channels = arr.shape[0]
            # Choose human‑readable channel names when possible
            if modality == "audio":
                ch_names = ["audio"]
            elif modality == "phone":
                # phone driver emits screen_on, tap_rate and orientation in that order
                ch_names = ["screen_on", "tap_rate", "orientation"][:channels]
            elif modality == "fnirs":
                # fNIRS optodes are labelled optode0, optode1, …
                ch_names = [f"optode{i}" for i in range(channels)]
            else:
                ch_names = [f"ch{i}" for i in range(channels)]
            write_nwb_file(
                file_path,
                data=[arr[i] for i in range(channels)],
                sampling_rate=fs,
                channel_names=ch_names,
                session_id=f"{session_id}_{modality}",
                subject_id=subject_id,
                device_name=modality,
                description=f"{modality} recording",
            )
            file_paths[modality] = file_path
        except Exception as exc:
            # Fallback: save channels as a NumPy .npz file when NWB is unavailable
            fallback_path = os.path.join(
                output_dir, f"{subject_id}_{session_id}_{modality}.npz"
            )
            np.savez(
                fallback_path,
                **{ch_names[i]: arr[i] for i in range(channels)},
                sampling_rate=fs,
            )
            file_paths[modality] = fallback_path
            logger.warning(
                "Failed to write %s data to NWB (error: %s). Saved fallback .npz file instead.",
                modality,
                exc,
            )
    # handle video modality (write OME‑Zarr)
    if "video" in collectors:
        collector = collectors["video"]
        frames = collector.data
        if frames:
            try:
                # frames list contains images with shape (height, width, channels)
                arr = np.stack(frames, axis=0).astype(np.float32)  # shape (t, h, w, c)
                # convert to 5D (t, c, z, y, x) expected by write_ome_zarr
                arr5 = np.transpose(arr, (0, 3, 2, 1))  # (t, c, y, x)
                arr5 = arr5[:, :, np.newaxis, :, :]  # add z=1 dimension
                zarr_path = os.path.join(output_dir, f"{subject_id}_{session_id}_video.zarr")
                write_ome_zarr(
                    array=arr5,
                    store_path=zarr_path,
                    chunk_shape=(1, arr5.shape[1], 1, min(arr5.shape[3], 64), min(arr5.shape[4], 64)),
                    compressor=None,
                    metadata={"modality": "video"},
                )
                file_paths["video"] = zarr_path
            except Exception as exc:
                # Fallback: save raw video frames as a NumPy array
                fallback_path = os.path.join(
                    output_dir, f"{subject_id}_{session_id}_video.npy"
                )
                np.save(fallback_path, arr)
                file_paths["video"] = fallback_path
                logger.warning(
                    "Failed to write video data to Zarr (error: %s). Saved fallback .npy file instead.",
                    exc,
                )
    return file_paths


def export_and_train(
    raw_dir: str,
    features_dir: str,
    job_name: str,
    sagemaker_config: Optional[str] = None,
) -> None:
    """Export aligned data to training shards and launch a training job.

    This function assumes that raw data has been aligned and curated into
    per‑modality directories (silver tier).  It uses the WebDataset
    exporter to shard the data into tar files under ``features_dir``
    (gold tier) and then launches a SageMaker DDP job using the
    provided configuration file.

    Parameters
    ----------
    raw_dir : str
        Directory containing curated sample directories or files.
    features_dir : str
        Output directory for WebDataset shards.
    job_name : str
        Name for the SageMaker training job.
    sagemaker_config : str, optional
        Path to a JSON or YAML file with SageMaker job configuration.
    """
    # Ensure the output directory exists even if the exporter does nothing
    os.makedirs(features_dir, exist_ok=True)
    # export to WebDataset shards
    try:
        export_to_webdataset(input_uri=raw_dir, output_uri=features_dir, shard_size=50)
    except Exception as exc:
        logger.error("Failed to export to WebDataset: %s", exc)
    # optionally export to Petastorm (not used by DDP training)
    # export_to_petastorm(raw_dir, features_dir + "_petastorm")
    # launch training (this will no‑op if sagemaker package missing)
    try:
        launch_training(job_name=job_name, config_path=sagemaker_config)
    except Exception as exc:
        logger.error("Failed to launch SageMaker training: %s", exc)


async def run_constellation_demo(
    duration: float = 10.0,
    kafka_bootstrap: Optional[str] = "localhost:9092",
    topic_prefix: str = "raw",
    subject_id: str = "demo_subject",
    session_id: str = "demo_session",
    output_base: str = "/tmp/constellation_demo",
    fault_injection: bool = False,
    sagemaker_config: Optional[str] = None,
) -> None:
    """Run a complete Constellation demo pipeline end‑to‑end.

    This coroutine orchestrates multi‑modal ingestion and streaming,
    writes raw data to disk (as a stand‑in for S3), exports features to
    WebDataset shards and kicks off a SageMaker training job.  The
    resulting metrics are logged and Prometheus metrics are updated.

    Parameters
    ----------
    duration : float, optional
        Number of seconds to ingest data.  Defaults to 10.
    kafka_bootstrap : Optional[str], optional
        Bootstrap servers for Kafka.  Defaults to "localhost:9092".  If
        ``None`` or an empty string, ingestion runs in dry‑run mode
        without a Kafka broker.
    topic_prefix : str, optional
        Prefix for Kafka topics.  Defaults to "raw".
    subject_id : str, optional
        Identifier for the subject.
    session_id : str, optional
        Identifier for the session.
    output_base : str, optional
        Base directory where raw and processed data are stored.  Defaults
        to ``/tmp/constellation_demo``.
    fault_injection : bool, optional
        Enable synthetic fault injection (packet drops, jitter).
    sagemaker_config : str, optional
        Path to a configuration file for the SageMaker training job.
    """
    collectors = await run_multimodal_ingestion(
        duration=duration,
        kafka_bootstrap=kafka_bootstrap,
        topic_prefix=topic_prefix,
        subject_id=subject_id,
        session_id=session_id,
        fault_injection=fault_injection,
    )
    # write raw data to disk
    raw_dir = os.path.join(output_base, "raw", subject_id, session_id)
    file_paths = write_raw_data_to_storage(collectors, raw_dir, subject_id, session_id)
    logger.info("Raw data written: %s", file_paths)
    # export to WebDataset and train model
    features_dir = os.path.join(output_base, "gold", subject_id, session_id)
    export_and_train(
        raw_dir,
        features_dir,
        job_name=f"{subject_id}-{session_id}-job",
        sagemaker_config=sagemaker_config,
    )
    logger.info("Constellation demo pipeline completed")


__all__ = [
    "SensorCollector",
    "run_multimodal_ingestion",
    "write_raw_data_to_storage",
    "export_and_train",
    "run_constellation_demo",
]