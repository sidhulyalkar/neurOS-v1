"""
Cloud storage abstractions for neurOS.

The :mod:`neuros.cloud` package provides simple interfaces for persisting
pipeline metrics and streaming results to external storage backends.  In
production a cloud storage provider such as Amazon S3 or Azure Blob
Storage would be used.  During development the :class:`LocalStorage`
backend can be used to write data to the local filesystem.

Classes
-------
CloudStorage
    Abstract base class defining the storage interface.

LocalStorage
    Concrete implementation that writes metrics to JSON files in a
    designated directory and appends streaming results to a log file.

S3Storage
    Optional implementation using ``boto3`` for uploading metrics to
    Amazon S3.  This class is only available if ``boto3`` is installed
    and AWS credentials are configured in the environment.

Usage
-----
Configure a storage backend and pass it to the API server or pipeline
wrapper to have metrics automatically uploaded at the end of a run.

Example::

    from neuros.cloud import LocalStorage
    storage = LocalStorage(base_dir="~/neuros_runs")
    # run pipeline and then upload metrics
    metrics = asyncio.run(pipeline.run(5.0))
    storage.upload_metrics(metrics, run_id="2023-01-01T00:00:00")

"""

from __future__ import annotations

import json
import os
import threading
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Tuple

__all__ = ["CloudStorage", "LocalStorage"]

try:
    import boto3  # type: ignore
    from botocore.exceptions import NoCredentialsError, ClientError  # type: ignore
    HAS_BOTO3 = True
except ImportError:  # pragma: no cover - boto3 is optional
    HAS_BOTO3 = False


class CloudStorage(ABC):
    """Abstract base class for storage backends."""

    @abstractmethod
    def upload_metrics(self, metrics: Dict[str, float], run_id: str) -> None:
        """Upload run metrics to the storage backend.

        Parameters
        ----------
        metrics : dict
            Dictionary containing performance metrics such as duration,
            throughput and accuracy.
        run_id : str
            Unique identifier for this run; used to group metrics and
            streaming logs in storage.
        """
        raise NotImplementedError

    @abstractmethod
    def stream_results(self, results: Iterable[Tuple[float, int, float, float]], run_id: str) -> None:
        """Persist streamed classification results.

        This method can be used to log real‑time classification results
        (timestamp, label, confidence, latency) as they are produced.  The
        implementation may choose to buffer writes for efficiency.

        Parameters
        ----------
        results : iterable
            Iterable of tuples ``(timestamp, label, confidence, latency)``.
        run_id : str
            Unique identifier for this run.
        """
        raise NotImplementedError

    def upload_database(self, db_path: str) -> None:
        """Upload the neurOS database file to the storage backend.

        This optional method allows persisting the entire SQLite
        database used by neurOS.  Implementations should handle any
        necessary serialization or transfer to ensure the database can
        be restored later.  By default this method does nothing.

        Parameters
        ----------
        db_path : str
            Path to the local database file.
        """
        return


class LocalStorage(CloudStorage):
    """Local filesystem implementation of :class:`CloudStorage`.

    Metrics are saved as JSON files under ``base_dir/run_id/metrics.json`` and
    streaming logs are appended to ``base_dir/run_id/stream.log``.
    """

    def __init__(self, base_dir: str | os.PathLike[str] = "~/neuros_runs") -> None:
        self.base_path = Path(os.path.expanduser(base_dir))
        self.base_path.mkdir(parents=True, exist_ok=True)
        # lock to serialize writes to the same run directory
        self._lock = threading.Lock()

    def _run_dir(self, run_id: str) -> Path:
        dir_path = self.base_path / run_id
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path

    def upload_metrics(self, metrics: Dict[str, float], run_id: str) -> None:
        """Persist metrics to the local filesystem with optional encryption.

        Metrics are saved in three forms:

        * ``metrics.json`` – human readable JSON
        * ``metrics.b64`` – base64 encoded JSON (simple obfuscation)
        * ``metrics.enc`` – encrypted JSON when ``NEUROS_ENCRYPTION_KEY`` is set.  Two
          encryption mechanisms are supported:

            1. **Fernet AES encryption** if the ``cryptography`` package is
               available.  The key provided via ``NEUROS_ENCRYPTION_KEY``
               is hashed to 32 bytes and base64‑encoded to form a
               Fernet key.  Data is encrypted and stored as a base64
               encoded string.
            2. **XOR cipher** fallback when ``cryptography`` is not
               installed.  The environment key is hashed and XORed
               against the plaintext bytes.  The result is base64
               encoded.

        If no encryption key is configured, only JSON and base64 files
        are written.
        """
        import base64
        import hashlib
        key = os.getenv("NEUROS_ENCRYPTION_KEY")
        # attempt to use cryptography for stronger encryption
        _fernet = None
        if key:
            try:
                from cryptography.fernet import Fernet  # type: ignore
                # derive 32‑byte key from input using SHA256 and then base64‑encode
                digest = hashlib.sha256(key.encode("utf-8")).digest()
                # Fernet requires a 32 byte key then base64 encoded (44 chars)
                fernet_key = base64.urlsafe_b64encode(digest)
                _fernet = Fernet(fernet_key)
            except Exception:
                _fernet = None
        with self._lock:
            run_dir = self._run_dir(run_id)
            json_str = json.dumps(metrics, indent=2)
            # write plain JSON
            (run_dir / "metrics.json").write_text(json_str, encoding="utf-8")
            # write base64 encoded JSON
            b64_encoded = base64.b64encode(json_str.encode("utf-8")).decode("ascii")
            (run_dir / "metrics.b64").write_text(b64_encoded, encoding="utf-8")
            # optional encrypted file
            if key:
                if _fernet is not None:
                    # encrypt using Fernet
                    enc_bytes = _fernet.encrypt(json_str.encode("utf-8"))
                    enc_b64 = enc_bytes.decode("ascii")
                else:
                    # fallback XOR cipher
                    digest = hashlib.sha256(key.encode("utf-8")).digest()
                    data_bytes = json_str.encode("utf-8")
                    xor_bytes = bytearray(len(data_bytes))
                    for i, b in enumerate(data_bytes):
                        xor_bytes[i] = b ^ digest[i % len(digest)]
                    enc_b64 = base64.b64encode(xor_bytes).decode("ascii")
                (run_dir / "metrics.enc").write_text(enc_b64, encoding="utf-8")

    def stream_results(self, results: Iterable[Tuple[float, int, float, float]], run_id: str) -> None:
        """Persist streaming results with optional encryption.

        Each streaming tuple (timestamp, label, confidence, latency) is
        written in three forms:

        * ``stream.log`` – plain tab‑separated text
        * ``stream.b64`` – base64 encoded form of each line
        * ``stream.enc`` – encrypted line when ``NEUROS_ENCRYPTION_KEY``
          is defined.  This method supports two encryption modes:

            1. ``cryptography.fernet.Fernet`` if available.  The key
               provided via ``NEUROS_ENCRYPTION_KEY`` is hashed to
               32 bytes and converted to a URL‑safe base64 key for
               Fernet.  Each line is encrypted via ``fernet.encrypt``
               and written as text.
            2. XOR cipher fallback when ``cryptography`` is unavailable.
               The hashed key is XORed against each byte of the line
               before base64 encoding.

        If no encryption key is configured, only ``stream.log`` and
        ``stream.b64`` are produced.
        """
        import base64
        import hashlib
        key = os.getenv("NEUROS_ENCRYPTION_KEY")
        # Determine encryption backend
        _fernet = None
        digest = None
        if key:
            try:
                from cryptography.fernet import Fernet  # type: ignore
                digest_sha = hashlib.sha256(key.encode("utf-8")).digest()
                fernet_key = base64.urlsafe_b64encode(digest_sha)
                _fernet = Fernet(fernet_key)
            except Exception:
                _fernet = None
                digest = hashlib.sha256(key.encode("utf-8")).digest()
        with self._lock:
            run_dir = self._run_dir(run_id)
            log_path = run_dir / "stream.log"
            b64_path = run_dir / "stream.b64"
            enc_path = run_dir / "stream.enc"
            with open(log_path, "a", encoding="utf-8") as f_log, open(b64_path, "a", encoding="utf-8") as f_b64:
                f_enc = open(enc_path, "a", encoding="utf-8") if key else None
                try:
                    for ts, label, conf, latency in results:
                        line = f"{ts:.3f}\t{label}\t{conf:.3f}\t{latency:.6f}\n"
                        # write plain log
                        f_log.write(line)
                        # write base64 obfuscation
                        b64_line = base64.b64encode(line.encode("utf-8")).decode("ascii")
                        f_b64.write(b64_line + "\n")
                        # write encrypted line if key provided
                        if key and f_enc:
                            if _fernet is not None:
                                enc_bytes = _fernet.encrypt(line.encode("utf-8"))
                                enc_line = enc_bytes.decode("ascii")
                            else:
                                # XOR fallback
                                data_bytes = line.encode("utf-8")
                                xor_bytes = bytearray(len(data_bytes))
                                for i, b in enumerate(data_bytes):
                                    xor_bytes[i] = b ^ digest[i % len(digest)]  # type: ignore[index]
                                enc_line = base64.b64encode(xor_bytes).decode("ascii")
                            f_enc.write(enc_line + "\n")
                finally:
                    if f_enc:
                        f_enc.close()

    def upload_database(self, db_path: str) -> None:
        """Backup the SQLite database to the local storage directory.

        For the local storage backend, copying the database file to the
        base directory suffices as a backup.  The file name is
        preserved.  Any errors during copying are silently ignored.
        """
        import shutil
        from pathlib import Path
        src = Path(db_path)
        if not src.is_file():
            return
        dst = self.base_path / src.name
        try:
            shutil.copy(src, dst)
        except Exception:
            pass


if HAS_BOTO3:
    class S3Storage(CloudStorage):  # pragma: no cover - boto3 is optional
        """Amazon S3 implementation of :class:`CloudStorage`.

        Parameters
        ----------
        bucket : str
            Name of the S3 bucket to write to.
        prefix : str
            Optional prefix to prepend to object keys.
        s3_client : boto3.client
            Optional boto3 S3 client instance.  If omitted, a client will
            be created with default credentials.
        """

        def __init__(self, bucket: str, *, prefix: str = "", s3_client: boto3.client | None = None) -> None:
            self.bucket = bucket
            self.prefix = prefix.rstrip("/") if prefix else ""
            self.s3 = s3_client or boto3.client("s3")

        def _object_key(self, run_id: str, filename: str) -> str:
            key = f"{run_id}/{filename}"
            return f"{self.prefix}/{key}" if self.prefix else key

        def upload_metrics(self, metrics: Dict[str, float], run_id: str) -> None:
            data = json.dumps(metrics).encode("utf-8")
            key = self._object_key(run_id, "metrics.json")
            try:
                self.s3.put_object(Bucket=self.bucket, Key=key, Body=data, ContentType="application/json")
            except (NoCredentialsError, ClientError) as exc:
                raise RuntimeError(f"Failed to upload metrics to S3: {exc}")

        def stream_results(self, results: Iterable[Tuple[float, int, float, float]], run_id: str) -> None:
            import io
            buffer = io.StringIO()
            for ts, label, conf, latency in results:
                buffer.write(f"{ts:.3f}\t{label}\t{conf:.3f}\t{latency:.6f}\n")
            key = self._object_key(run_id, "stream.log")
            try:
                self.s3.put_object(Bucket=self.bucket, Key=key, Body=buffer.getvalue().encode("utf-8"), ContentType="text/plain")
            except (NoCredentialsError, ClientError) as exc:
                raise RuntimeError(f"Failed to upload stream results to S3: {exc}")

        def upload_database(self, db_path: str) -> None:
            """Upload the SQLite database to S3.

            The database file is uploaded to the configured bucket under
            a key equal to its filename, optionally prefixed by the
            storage prefix.  This allows backing up the entire database
            for disaster recovery.
            """
            import os
            import boto3  # type: ignore
            from botocore.exceptions import NoCredentialsError, ClientError  # type: ignore
            key = self._object_key("", os.path.basename(db_path))
            try:
                with open(db_path, "rb") as f:
                    self.s3.put_object(
                        Bucket=self.bucket,
                        Key=key,
                        Body=f.read(),
                        ContentType="application/x-sqlite3",
                    )
            except (NoCredentialsError, ClientError) as exc:
                raise RuntimeError(f"Failed to upload database to S3: {exc}")

        def upload_database(self, db_path: str) -> None:
            """Upload the entire SQLite database to S3.

            This method uploads the given database file to the configured
            bucket and prefix.  The object key is the base name of the
            database file.
            """
            import os
            key = self._object_key("", os.path.basename(db_path))
            try:
                with open(db_path, "rb") as f:
                    self.s3.put_object(
                        Bucket=self.bucket,
                        Key=key,
                        Body=f,
                        ContentType="application/x-sqlite3",
                    )
            except (NoCredentialsError, ClientError) as exc:
                raise RuntimeError(f"Failed to upload database to S3: {exc}")
