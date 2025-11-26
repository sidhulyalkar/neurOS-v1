"""
Mechanistic Interpretability Database for Result Storage and Caching.

Provides efficient storage, retrieval, and caching of mechanistic interpretability
results across all analysis methods. Enables systematic comparisons, provenance
tracking, and workflow optimization.

Key Features:
- Content-based caching using SHA256 hashes (via MechIntResult.get_content_hash)
- Efficient HDF5 storage for large arrays (delegated to MechIntResult.save/load)
- SQLite metadata index for fast queries
- Provenance tracking across analysis chains
- Batch operations and parallel queries
- Automatic deduplication + cache cleanup
- In-memory result cache for fast reuse
- Basic versioning helpers and tag utilities
- Optional thumbnail support via metadata

Example:
    >>> db = MechIntDatabase("./mech_int_results")
    >>>
    >>> # Store result
    >>> result_id = db.store(my_result, tags=["sae", "layer3", "experiment1"])
    >>>
    >>> # Retrieve by ID
    >>> retrieved = db.get(result_id)
    >>>
    >>> # Or via alias:
    >>> retrieved = db.load(result_id)
    >>>
    >>> # Query by metadata
    >>> sae_results = db.query(method="SAE", tags=["experiment1"])
    >>>
    >>> # Find similar
    >>> similar = db.find_similar(my_result, threshold=0.9)

Author: NeuroS Team
Date: 2025-10-30 (updated)
"""

import sqlite3
import h5py  # kept for future direct HDF5 ops, even though MechIntResult handles most IO
import json
import hashlib
import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any, Set
from datetime import datetime
import numpy as np
import torch
from dataclasses import asdict
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import OrderedDict

from neuros_mechint.results import (
    MechIntResult,
    CircuitResult,
    DynamicsResult,
    InformationResult,
    AlignmentResult,
    FractalResult,
    ResultCollection,
)
from neuros_mechint.results_extended import (
    BiophysicalResult,
    InterventionResult,
    CriticalityResult,
    MultifractalResult,
)

logger = logging.getLogger(__name__)


class MechIntDatabase:
    """
    Database for storing and retrieving mechanistic interpretability results.

    Uses SQLite for metadata indexing and HDF5 for array storage.
    Implements content-based caching for automatic deduplication.

    Args:
        root_dir: Root directory for database storage
        auto_cache: Automatically cache results based on content hash
        max_cache_size_gb: Maximum on-disk cache size in GB (default: 10)
        compression: HDF5 compression level 0-9 (default: 4)
        verbose: Enable verbose logging
        memory_cache_size: Max number of results to keep in in-memory LRU cache
    """

    def __init__(
        self,
        root_dir: Union[str, Path],
        auto_cache: bool = True,
        max_cache_size_gb: float = 10.0,
        compression: int = 4,
        verbose: bool = True,
        memory_cache_size: int = 256,
    ):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

        self.auto_cache = auto_cache
        self.max_cache_size_gb = max_cache_size_gb
        self.compression = compression
        self.verbose = verbose

        # Storage directories
        self.data_dir = self.root_dir / "data"
        self.data_dir.mkdir(exist_ok=True)

        # Optional thumbnails directory
        self.thumbnails_dir = self.root_dir / "thumbnails"
        self.thumbnails_dir.mkdir(exist_ok=True)

        # Metadata DB path
        self.metadata_db = self.root_dir / "metadata.db"

        # In-memory LRU-style cache: result_id -> MechIntResult
        self._memory_cache: "OrderedDict[str, MechIntResult]" = OrderedDict()
        self._memory_cache_size = max(0, int(memory_cache_size))

        # Initialize database schema
        self._init_database()

        self._log(f"Initialized MechIntDatabase at {self.root_dir}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log(self, message: str):
        """Log message if verbose."""
        if self.verbose:
            logger.info(f"[MechIntDatabase] {message}")

    def _init_database(self):
        """Initialize SQLite database schema."""
        conn = sqlite3.connect(self.metadata_db)
        cursor = conn.cursor()

        # Main results table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS results (
                result_id TEXT PRIMARY KEY,
                content_hash TEXT UNIQUE,
                method TEXT,
                result_type TEXT,
                timestamp TEXT,
                file_path TEXT,
                size_bytes INTEGER,
                metadata_json TEXT,
                metrics_json TEXT
            )
        """)

        # Tags table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tags (
                result_id TEXT,
                tag TEXT,
                FOREIGN KEY (result_id) REFERENCES results(result_id),
                PRIMARY KEY (result_id, tag)
            )
        """)

        # Provenance table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS provenance (
                result_id TEXT,
                parent_id TEXT,
                FOREIGN KEY (result_id) REFERENCES results(result_id),
                FOREIGN KEY (parent_id) REFERENCES results(result_id),
                PRIMARY KEY (result_id, parent_id)
            )
        """)

        # Create indices
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_method ON results(method)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_type ON results(result_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON results(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_hash ON results(content_hash)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tag ON tags(tag)")

        conn.commit()
        conn.close()

    def _update_memory_cache(self, result_id: str, result: MechIntResult):
        """Insert/refresh an entry in the in-memory LRU cache."""
        if self._memory_cache_size <= 0:
            return

        if result_id in self._memory_cache:
            # Move to end (most recently used)
            self._memory_cache.move_to_end(result_id)
        else:
            self._memory_cache[result_id] = result
            # Evict least-recently used if over limit
            if len(self._memory_cache) > self._memory_cache_size:
                evicted_id, _ = self._memory_cache.popitem(last=False)
                self._log(f"Evicted {evicted_id} from in-memory cache")

    def _get_from_memory_cache(self, result_id: str) -> Optional[MechIntResult]:
        """Retrieve an entry from the in-memory cache (if present)."""
        result = self._memory_cache.get(result_id)
        if result is not None:
            # Mark as recently used
            self._memory_cache.move_to_end(result_id)
        return result

    def _get_by_hash(self, content_hash: str) -> Optional[str]:
        """Get result ID by content hash."""
        conn = sqlite3.connect(self.metadata_db)
        cursor = conn.cursor()
        cursor.execute("SELECT result_id FROM results WHERE content_hash = ?", (content_hash,))
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else None

    # ------------------------------------------------------------------
    # Core store/load APIs
    # ------------------------------------------------------------------

    def store(
        self,
        result: MechIntResult,
        tags: Optional[List[str]] = None,
        result_id: Optional[str] = None
    ) -> str:
        """
        Store a result in the database.

        Args:
            result: MechIntResult (or subclass) to store
            tags: Optional tags for categorization
            result_id: Optional custom ID (default: auto-generated)

        Returns:
            result_id: Unique identifier for the stored result
        """
        # Generate content hash (reproducibility & dedup)
        content_hash = result.get_content_hash()

        # Check if already cached
        if self.auto_cache:
            existing_id = self._get_by_hash(content_hash)
            if existing_id:
                self._log(f"Result already cached with ID: {existing_id}")
                return existing_id

        # Generate result ID
        if result_id is None:
            result_id = f"{result.method}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

        # Store data to HDF5 (single .h5 file per result)
        file_path = self.data_dir / f"{result_id}.h5"
        result.save(str(file_path))

        # Maybe save thumbnail (if result exposes one)
        thumb_rel_path = self._maybe_save_thumbnail(result, result_id)

        # Merge thumbnail info into metadata if present
        metadata = dict(result.metadata or {})
        if thumb_rel_path is not None:
            metadata["_thumbnail_path"] = str(thumb_rel_path)

        # Get file size
        size_bytes = file_path.stat().st_size

        # Store metadata in SQLite
        conn = sqlite3.connect(self.metadata_db)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO results (result_id, content_hash, method, result_type,
                               timestamp, file_path, size_bytes, metadata_json, metrics_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            result_id,
            content_hash,
            result.method,
            type(result).__name__,
            result.timestamp or datetime.now().isoformat(),
            str(file_path),
            size_bytes,
            json.dumps(metadata),
            json.dumps(result.metrics or {}),
        ))

        # Store tags
        if tags:
            for tag in tags:
                cursor.execute(
                    "INSERT INTO tags (result_id, tag) VALUES (?, ?)",
                    (result_id, tag)
                )

        # Store provenance
        if getattr(result, "provenance", None):
            for parent in result.provenance:
                try:
                    parent_hash = parent.get_content_hash()
                    parent_id = self._get_by_hash(parent_hash)
                    if parent_id:
                        cursor.execute(
                            "INSERT INTO provenance (result_id, parent_id) VALUES (?, ?)",
                            (result_id, parent_id)
                        )
                except Exception as e:
                    logger.error(f"Error storing provenance for {result_id}: {e}")

        conn.commit()
        conn.close()

        self._log(f"Stored result with ID: {result_id}")

        # Update in-memory cache
        self._update_memory_cache(result_id, result)

        # Check cache size and cleanup if needed
        if self.auto_cache:
            self._cleanup_cache()

        return result_id

    def get(self, result_id: str) -> Optional[MechIntResult]:
        """
        Retrieve a result by ID. Returns None if not found.

        Uses in-memory cache if available, otherwise loads from HDF5.
        """
        # Check in-memory cache first
        cached = self._get_from_memory_cache(result_id)
        if cached is not None:
            return cached

        conn = sqlite3.connect(self.metadata_db)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT file_path, result_type, metadata_json, metrics_json FROM results WHERE result_id = ?",
            (result_id,)
        )
        row = cursor.fetchone()
        conn.close()

        if not row:
            self._log(f"Result not found: {result_id}")
            return None

        file_path, result_type, metadata_json, metrics_json = row

        # Select appropriate result class
        result_class = {
            'MechIntResult': MechIntResult,
            'CircuitResult': CircuitResult,
            'DynamicsResult': DynamicsResult,
            'InformationResult': InformationResult,
            'AlignmentResult': AlignmentResult,
            'FractalResult': FractalResult,
            'BiophysicalResult': BiophysicalResult,
            'InterventionResult': InterventionResult,
            'CriticalityResult': CriticalityResult,
            'MultifractalResult': MultifractalResult,
        }.get(result_type, MechIntResult)

        result = result_class.load(file_path)

        # Restore metadata/metrics from DB (in case they changed)
        try:
            result.metadata = json.loads(metadata_json) if metadata_json else {}
        except Exception:
            result.metadata = {}
        try:
            result.metrics = json.loads(metrics_json) if metrics_json else {}
        except Exception:
            result.metrics = {}

        # Update in-memory cache
        self._update_memory_cache(result_id, result)

        return result

    def load(self, result_id: str) -> MechIntResult:
        """
        Alias for `get`, but raises KeyError if the result is not found.

        This is useful for components (like CircuitComparator) that expect
        failures to be exceptional rather than returning None.
        """
        result = self.get(result_id)
        if result is None:
            raise KeyError(f"No result with ID {result_id} found in MechIntDatabase.")
        return result

    # ------------------------------------------------------------------
    # Thumbnail helpers
    # ------------------------------------------------------------------

    def _maybe_save_thumbnail(self, result: MechIntResult, result_id: str) -> Optional[Path]:
        """
        Optionally save a thumbnail for the result, if supported.

        Protocol:
        - If `result` exposes a `.get_thumbnail()` method:
            - It may return:
                - a numpy array (H x W or H x W x 3/4)
                - a path to an existing image file
        - We save a PNG (or copy) into root_dir/thumbnails/<result_id>.png
        - The relative path is then stored in metadata as `_thumbnail_path`.

        Returns:
            Relative Path to thumbnail file (relative to root_dir), or None
        """
        if not hasattr(result, "get_thumbnail"):
            return None

        try:
            thumb = result.get_thumbnail()
        except Exception as e:
            logger.error(f"Error generating thumbnail for {result_id}: {e}")
            return None

        if thumb is None:
            return None

        thumb_path = self.thumbnails_dir / f"{result_id}.png"

        try:
            # Case 1: numpy array
            if isinstance(thumb, np.ndarray):
                import imageio.v2 as imageio  # lazy import
                imageio.imwrite(thumb_path, thumb)
            # Case 2: path-like (string or Path)
            elif isinstance(thumb, (str, Path)):
                src = Path(thumb)
                if src.is_file():
                    import shutil
                    shutil.copy(src, thumb_path)
                else:
                    return None
            else:
                # Unsupported type
                return None
        except Exception as e:
            logger.error(f"Failed to save thumbnail for {result_id}: {e}")
            return None

        # Return path relative to root_dir
        return thumb_path.relative_to(self.root_dir)

    def get_thumbnail_path(self, result_id: str) -> Optional[Path]:
        """
        Return the thumbnail path for a given result, if present in metadata.
        """
        conn = sqlite3.connect(self.metadata_db)
        cursor = conn.cursor()
        cursor.execute("SELECT metadata_json FROM results WHERE result_id = ?", (result_id,))
        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        try:
            meta = json.loads(row[0]) if row[0] else {}
        except Exception:
            meta = {}

        thumb_rel = meta.get("_thumbnail_path")
        if not thumb_rel:
            return None

        return self.root_dir / thumb_rel

    # ------------------------------------------------------------------
    # Querying and tag utilities
    # ------------------------------------------------------------------

    def query(
        self,
        method: Optional[str] = None,
        result_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[str]:
        """
        General query by metadata criteria.

        Args:
            method: Filter by method name
            result_type: Filter by result type class name
            tags: Filter by tags (all must match)
            start_time: Filter by timestamp >= start_time (ISO8601 string)
            end_time: Filter by timestamp <= end_time (ISO8601 string)
            limit: Maximum number of results

        Returns:
            List of result IDs matching criteria
        """
        conn = sqlite3.connect(self.metadata_db)
        cursor = conn.cursor()

        query = "SELECT DISTINCT r.result_id FROM results r"
        conditions = []
        params: List[Any] = []

        # Join with tags if needed
        if tags:
            query += " INNER JOIN tags t ON r.result_id = t.result_id"

        # Add conditions
        if method:
            conditions.append("r.method = ?")
            params.append(method)

        if result_type:
            conditions.append("r.result_type = ?")
            params.append(result_type)

        if start_time:
            conditions.append("r.timestamp >= ?")
            params.append(start_time)

        if end_time:
            conditions.append("r.timestamp <= ?")
            params.append(end_time)

        if tags:
            # All tags must match
            tag_conditions = " OR ".join(["t.tag = ?"] * len(tags))
            conditions.append(f"({tag_conditions})")
            params.extend(tags)
            query += f" GROUP BY r.result_id HAVING COUNT(DISTINCT t.tag) = {len(tags)}"

        # Build query
        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY r.timestamp DESC"

        if limit:
            query += f" LIMIT {limit}"

        cursor.execute(query, params)
        result_ids = [row[0] for row in cursor.fetchall()]

        conn.close()

        self._log(f"Query returned {len(result_ids)} results")
        return result_ids

    def search_by_tags(
        self,
        include_tags: Optional[List[str]] = None,
        exclude_tags: Optional[List[str]] = None,
        limit: Optional[int] = None
    ) -> List[str]:
        """
        Convenience tag search.

        Args:
            include_tags: Results must have ALL of these tags
            exclude_tags: Results must have NONE of these tags
            limit: Maximum number of results

        Returns:
            List of result IDs
        """
        include_tags = include_tags or []
        exclude_tags = exclude_tags or []

        conn = sqlite3.connect(self.metadata_db)
        cursor = conn.cursor()

        query = "SELECT DISTINCT r.result_id FROM results r"
        params: List[Any] = []

        if include_tags:
            query += " INNER JOIN tags t_in ON r.result_id = t_in.result_id"
        if exclude_tags:
            query += " LEFT JOIN tags t_ex ON r.result_id = t_ex.result_id"

        conditions: List[str] = []

        if include_tags:
            tag_in_cond = " OR ".join(["t_in.tag = ?"] * len(include_tags))
            conditions.append(f"({tag_in_cond})")
            params.extend(include_tags)
            query += f" GROUP BY r.result_id HAVING COUNT(DISTINCT t_in.tag) = {len(include_tags)}"

        if exclude_tags:
            # Exclusion via NOT IN subquery is simpler
            pass

        # Build WHERE for exclusion
        if exclude_tags:
            if not conditions:
                query += " WHERE "
            else:
                query += " AND "
            placeholders = ", ".join(["?"] * len(exclude_tags))
            query += f"r.result_id NOT IN (SELECT result_id FROM tags WHERE tag IN ({placeholders}))"
            params.extend(exclude_tags)

        query += " ORDER BY r.timestamp DESC"
        if limit:
            query += f" LIMIT {limit}"

        cursor.execute(query, params)
        result_ids = [row[0] for row in cursor.fetchall()]
        conn.close()

        return result_ids

    def get_tags(self, result_id: str) -> List[str]:
        """Get all tags for a result."""
        conn = sqlite3.connect(self.metadata_db)
        cursor = conn.cursor()
        cursor.execute("SELECT tag FROM tags WHERE result_id = ?", (result_id,))
        tags = [row[0] for row in cursor.fetchall()]
        conn.close()
        return tags

    def add_tags(self, result_id: str, tags: List[str]):
        """Add tags to a result."""
        conn = sqlite3.connect(self.metadata_db)
        cursor = conn.cursor()
        for tag in tags:
            cursor.execute(
                "INSERT OR IGNORE INTO tags (result_id, tag) VALUES (?, ?)",
                (result_id, tag)
            )
        conn.commit()
        conn.close()

    # ------------------------------------------------------------------
    # Provenance, similarity, batch ops
    # ------------------------------------------------------------------

    def get_provenance(self, result_id: str) -> List[str]:
        """Get parent result IDs in provenance chain."""
        conn = sqlite3.connect(self.metadata_db)
        cursor = conn.cursor()
        cursor.execute("SELECT parent_id FROM provenance WHERE result_id = ?", (result_id,))
        parent_ids = [row[0] for row in cursor.fetchall()]
        conn.close()
        return parent_ids

    def find_similar(
        self,
        result: MechIntResult,
        threshold: float = 0.9,
        method: Optional[str] = None
    ) -> List[Tuple[str, float]]:
        """
        Find similar results based on comparison metrics.

        Args:
            result: Reference result
            threshold: Similarity threshold (0-1)
            method: Optionally filter by method

        Returns:
            List of (result_id, similarity_score) tuples
        """
        # Get candidates
        candidates = self.query(method=method or result.method)

        similar: List[Tuple[str, float]] = []
        for candidate_id in candidates:
            candidate = self.get(candidate_id)
            if candidate:
                comparison = result.compare(candidate)
                # Expect comparison to return a dict of metrics
                if comparison:
                    similarity = float(np.mean(list(comparison.values())))
                    if similarity >= threshold:
                        similar.append((candidate_id, similarity))

        # Sort by similarity
        similar.sort(key=lambda x: x[1], reverse=True)

        return similar

    def batch_get(self, result_ids: List[str], num_workers: int = 4) -> Dict[str, MechIntResult]:
        """
        Retrieve multiple results in parallel.

        Args:
            result_ids: List of result IDs
            num_workers: Number of parallel workers

        Returns:
            Dictionary mapping result_id to MechIntResult
        """
        results: Dict[str, MechIntResult] = {}

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_id = {executor.submit(self.get, rid): rid for rid in result_ids}

            for future in as_completed(future_to_id):
                rid = future_to_id[future]
                try:
                    result = future.result()
                    if result:
                        results[rid] = result
                except Exception as e:
                    logger.error(f"Error loading {rid}: {e}")

        return results

    # ------------------------------------------------------------------
    # Deletion, stats, cache cleanup
    # ------------------------------------------------------------------

    def delete(self, result_id: str):
        """Delete a result and its associated data."""
        conn = sqlite3.connect(self.metadata_db)
        cursor = conn.cursor()

        # Get file path
        cursor.execute("SELECT file_path FROM results WHERE result_id = ?", (result_id,))
        row = cursor.fetchone()

        if row:
            file_path = Path(row[0])

            # Delete file
            if file_path.exists():
                file_path.unlink()

            # Delete thumbnail if exists
            thumb_path = self.get_thumbnail_path(result_id)
            if thumb_path is not None and thumb_path.exists():
                thumb_path.unlink()

            # Delete from database
            cursor.execute("DELETE FROM tags WHERE result_id = ?", (result_id,))
            cursor.execute("DELETE FROM provenance WHERE result_id = ? OR parent_id = ?",
                           (result_id, result_id))
            cursor.execute("DELETE FROM results WHERE result_id = ?", (result_id,))

            conn.commit()
            self._log(f"Deleted result: {result_id}")

        conn.close()

        # Remove from in-memory cache if present
        if result_id in self._memory_cache:
            self._memory_cache.pop(result_id, None)

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        conn = sqlite3.connect(self.metadata_db)
        cursor = conn.cursor()

        # Count results
        cursor.execute("SELECT COUNT(*) FROM results")
        total_results = cursor.fetchone()[0]

        # Count by method
        cursor.execute("SELECT method, COUNT(*) FROM results GROUP BY method")
        by_method = dict(cursor.fetchall())

        # Count by type
        cursor.execute("SELECT result_type, COUNT(*) FROM results GROUP BY result_type")
        by_type = dict(cursor.fetchall())

        # Total size
        cursor.execute("SELECT SUM(size_bytes) FROM results")
        total_size = cursor.fetchone()[0] or 0

        # Most common tags
        cursor.execute("SELECT tag, COUNT(*) as cnt FROM tags GROUP BY tag ORDER BY cnt DESC LIMIT 10")
        top_tags = dict(cursor.fetchall())

        conn.close()

        return {
            'total_results': total_results,
            'by_method': by_method,
            'by_type': by_type,
            'total_size_gb': total_size / (1024**3),
            'top_tags': top_tags
        }

    def count(self) -> int:
        """Convenience method: return total number of results."""
        return self.get_stats()["total_results"]

    def _cleanup_cache(self):
        """Remove oldest results if on-disk cache exceeds size limit."""
        stats = self.get_stats()

        if stats['total_size_gb'] > self.max_cache_size_gb:
            # Calculate how much to delete
            to_delete_gb = stats['total_size_gb'] - self.max_cache_size_gb * 0.8  # keep 80% of limit

            conn = sqlite3.connect(self.metadata_db)
            cursor = conn.cursor()

            # Get oldest results
            cursor.execute("""
                SELECT result_id, size_bytes
                FROM results
                ORDER BY timestamp ASC
            """)

            deleted_size = 0
            to_delete_gb_bytes = to_delete_gb * (1024**3)

            for result_id, size_bytes in cursor.fetchall():
                if deleted_size >= to_delete_gb_bytes:
                    break

                self.delete(result_id)
                deleted_size += size_bytes

            conn.close()

            self._log(f"Cleaned up {deleted_size / (1024**3):.2f} GB from cache")

    # ------------------------------------------------------------------
    # Collections, specialized queries, and summaries
    # ------------------------------------------------------------------

    def export_collection(
        self,
        result_ids: List[str],
        output_path: Union[str, Path],
        name: Optional[str] = None
    ) -> ResultCollection:
        """
        Export multiple results as a ResultCollection.

        Args:
            result_ids: List of result IDs
            output_path: Path to save collection
            name: Optional name for collection

        Returns:
            ResultCollection object
        """
        results = [self.get(rid) for rid in result_ids]
        results = [r for r in results if r is not None]

        collection = ResultCollection(
            results=results,
            name=name or f"collection_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        collection.save(str(output_path))
        self._log(f"Exported collection with {len(results)} results to {output_path}")

        return collection

    def query_biophysical(
        self,
        neuron_type: Optional[str] = None,
        has_metabolic: bool = False,
        limit: Optional[int] = None
    ) -> List[str]:
        """
        Query biophysical modeling results.

        Args:
            neuron_type: Filter by neuron type (e.g., 'pyramidal', 'interneuron')
            has_metabolic: Only return results with metabolic data
            limit: Maximum number of results

        Returns:
            List of result IDs
        """
        conn = sqlite3.connect(self.metadata_db)
        cursor = conn.cursor()

        query = "SELECT result_id FROM results WHERE result_type = 'BiophysicalResult'"
        params: List[Any] = []

        if neuron_type:
            query += " AND metadata_json LIKE ?"
            params.append(f'%"neuron_type": "{neuron_type}"%')

        if has_metabolic:
            query += " AND metadata_json LIKE ?"
            params.append('%"atp_levels"%')

        query += " ORDER BY timestamp DESC"

        if limit:
            query += f" LIMIT {limit}"

        cursor.execute(query, params)
        result_ids = [row[0] for row in cursor.fetchall()]
        conn.close()

        return result_ids

    def query_interventions(
        self,
        intervention_type: Optional[str] = None,
        min_effect_size: Optional[float] = None,
        limit: Optional[int] = None
    ) -> List[str]:
        """
        Query intervention results.

        Args:
            intervention_type: Filter by type ('optogenetics', 'pharmacology', 'stimulation')
            min_effect_size: Minimum effect size (currently not deeply parsed)
            limit: Maximum number of results

        Returns:
            List of result IDs
        """
        conn = sqlite3.connect(self.metadata_db)
        cursor = conn.cursor()

        query = "SELECT result_id FROM results WHERE result_type = 'InterventionResult'"
        params: List[Any] = []

        if intervention_type:
            query += " AND metadata_json LIKE ?"
            params.append(f'%"intervention_type": "{intervention_type}"%')

        query += " ORDER BY timestamp DESC"

        if limit:
            query += f" LIMIT {limit}"

        cursor.execute(query, params)
        result_ids = [row[0] for row in cursor.fetchall()]
        conn.close()

        return result_ids

    def query_criticality(
        self,
        near_critical: bool = False,
        threshold: float = 0.1,
        limit: Optional[int] = None
    ) -> List[str]:
        """
        Query criticality analysis results.

        Args:
            near_critical: Only return results near criticality (|σ - 1| < threshold)
            threshold: Threshold for criticality proximity
            limit: Maximum number of results

        Returns:
            List of result IDs
        """
        conn = sqlite3.connect(self.metadata_db)
        cursor = conn.cursor()

        query = "SELECT result_id, metrics_json FROM results WHERE result_type = 'CriticalityResult'"

        cursor.execute(query)
        rows = cursor.fetchall()

        result_ids: List[str] = []
        for result_id, metrics_json in rows:
            if near_critical:
                try:
                    metrics = json.loads(metrics_json or "{}")
                    branching = metrics.get('branching_parameter', 0)
                    if abs(branching - 1.0) < threshold:
                        result_ids.append(result_id)
                except Exception:
                    continue
            else:
                result_ids.append(result_id)

            if limit and len(result_ids) >= limit:
                break

        conn.close()

        self._log(f"Found {len(result_ids)} criticality results")
        return result_ids

    def query_multifractal(
        self,
        analysis_method: Optional[str] = None,
        is_multifractal: Optional[bool] = None,
        min_width: Optional[float] = None,
        limit: Optional[int] = None
    ) -> List[str]:
        """
        Query multifractal analysis results.

        Args:
            analysis_method: Filter by method ('mfdfa', 'wtmm')
            is_multifractal: Filter by multifractality
            min_width: Minimum multifractal width Δα
            limit: Maximum number of results

        Returns:
            List of result IDs
        """
        conn = sqlite3.connect(self.metadata_db)
        cursor = conn.cursor()

        query = "SELECT result_id, metadata_json, metrics_json FROM results WHERE result_type = 'MultifractalResult'"

        cursor.execute(query)
        rows = cursor.fetchall()

        result_ids: List[str] = []
        for result_id, metadata_json, metrics_json in rows:
            try:
                metadata = json.loads(metadata_json or "{}")
                metrics = json.loads(metrics_json or "{}")

                # Check analysis method
                if analysis_method and metadata.get('analysis_method') != analysis_method:
                    continue

                # Check multifractality
                if is_multifractal is not None:
                    if metadata.get('is_multifractal') != is_multifractal:
                        continue

                # Check width
                if min_width is not None:
                    width = metrics.get('multifractal_width', 0)
                    if width < min_width:
                        continue

                result_ids.append(result_id)

                if limit and len(result_ids) >= limit:
                    break

            except Exception:
                continue

        conn.close()

        self._log(f"Found {len(result_ids)} multifractal results")
        return result_ids

    def get_analysis_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary of all analyses in database.

        Returns:
            Dictionary with statistics for each analysis type
        """
        stats = self.get_stats()

        summary: Dict[str, Any] = {
            'total_results': stats['total_results'],
            'total_size_gb': stats['total_size_gb'],
            'by_method': stats['by_method'],
            'by_type': stats['by_type'],
        }

        conn = sqlite3.connect(self.metadata_db)
        cursor = conn.cursor()

        # Biophysical: count with metabolic data
        cursor.execute("""
            SELECT COUNT(*) FROM results
            WHERE result_type = 'BiophysicalResult'
            AND metadata_json LIKE '%atp_levels%'
        """)
        summary['biophysical_with_metabolic'] = cursor.fetchone()[0]

        # Interventions: count by type
        for int_type in ['optogenetics', 'pharmacology', 'stimulation']:
            cursor.execute(f"""
                SELECT COUNT(*) FROM results
                WHERE result_type = 'InterventionResult'
                AND metadata_json LIKE '%"intervention_type": "{int_type}"%'
            """)
            summary[f'interventions_{int_type}'] = cursor.fetchone()[0]

        # Criticality: count near critical
        cursor.execute("SELECT metrics_json FROM results WHERE result_type = 'CriticalityResult'")
        near_critical_count = 0
        for row in cursor.fetchall():
            try:
                metrics = json.loads(row[0] or "{}")
                if abs(metrics.get('branching_parameter', 0) - 1.0) < 0.1:
                    near_critical_count += 1
            except Exception:
                pass
        summary['criticality_near_critical'] = near_critical_count

        # Multifractal: count confirmed multifractal
        cursor.execute("""
            SELECT COUNT(*) FROM results
            WHERE result_type = 'MultifractalResult'
            AND metadata_json LIKE '%"is_multifractal": true%'
        """)
        summary['multifractal_confirmed'] = cursor.fetchone()[0]

        conn.close()

        return summary

    # ------------------------------------------------------------------
    # Versioning & hashing helpers
    # ------------------------------------------------------------------

    def get_hash(self, result_id: str) -> Optional[str]:
        """Return the content hash associated with a given result ID."""
        conn = sqlite3.connect(self.metadata_db)
        cursor = conn.cursor()
        cursor.execute("SELECT content_hash FROM results WHERE result_id = ?", (result_id,))
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else None

    def get_versions_by_hash(self, content_hash: str) -> List[str]:
        """
        Return all result IDs sharing the same content hash (should usually be 0 or 1
        unless you deliberately duplicated content with different IDs).
        """
        conn = sqlite3.connect(self.metadata_db)
        cursor = conn.cursor()
        cursor.execute("SELECT result_id FROM results WHERE content_hash = ? ORDER BY timestamp ASC", (content_hash,))
        ids = [row[0] for row in cursor.fetchall()]
        conn.close()
        return ids

    def get_versions_by_method_and_tags(
        self,
        method: str,
        tags: Optional[List[str]] = None
    ) -> List[str]:
        """
        Return all results for a given method (and optional tags) ordered by time.
        Useful as a lightweight 'version history' for a specific analysis pipeline.
        """
        ids = self.query(method=method, tags=tags)
        # Already ordered by timestamp DESC; flip to ASC for chronological order
        return list(reversed(ids))


__all__ = [
    'MechIntDatabase',
]
