"""
Mechanistic Interpretability Database for Result Storage and Caching.

Provides efficient storage, retrieval, and caching of mechanistic interpretability
results across all analysis methods. Enables systematic comparisons, provenance
tracking, and workflow optimization.

Key Features:
- Content-based caching using SHA256 hashes
- Efficient HDF5 storage for large arrays
- SQLite metadata index for fast queries
- Provenance tracking across analysis chains
- Batch operations and parallel queries
- Automatic deduplication
- Time-based versioning

Example:
    >>> db = MechIntDatabase("./mech_int_results")
    >>>
    >>> # Store result
    >>> result_id = db.store(my_result, tags=["sae", "layer3", "experiment1"])
    >>>
    >>> # Retrieve by ID
    >>> retrieved = db.get(result_id)
    >>>
    >>> # Query by metadata
    >>> sae_results = db.query(method="SAE", tags=["experiment1"])
    >>>
    >>> # Compare similar results
    >>> similar = db.find_similar(my_result, threshold=0.9)

Author: NeuroS Team
Date: 2025-10-30
"""

import sqlite3
import h5py
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any, Set
from datetime import datetime
import numpy as np
import torch
from dataclasses import asdict
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

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
        max_cache_size_gb: Maximum cache size in GB (default: 10)
        compression: HDF5 compression level 0-9 (default: 4)
        verbose: Enable verbose logging
    """

    def __init__(
        self,
        root_dir: Union[str, Path],
        auto_cache: bool = True,
        max_cache_size_gb: float = 10.0,
        compression: int = 4,
        verbose: bool = True
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

        self.metadata_db = self.root_dir / "metadata.db"

        # Initialize database
        self._init_database()

        self._log(f"Initialized MechIntDatabase at {self.root_dir}")

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

    def store(
        self,
        result: MechIntResult,
        tags: Optional[List[str]] = None,
        result_id: Optional[str] = None
    ) -> str:
        """
        Store a result in the database.

        Args:
            result: MechIntResult to store
            tags: Optional tags for categorization
            result_id: Optional custom ID (default: auto-generated)

        Returns:
            result_id: Unique identifier for the stored result
        """
        # Generate content hash
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

        # Store data to HDF5
        file_path = self.data_dir / f"{result_id}.h5"
        result.save(str(file_path))

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
            json.dumps(result.metadata),
            json.dumps(result.metrics)
        ))

        # Store tags
        if tags:
            for tag in tags:
                cursor.execute("INSERT INTO tags (result_id, tag) VALUES (?, ?)", (result_id, tag))

        # Store provenance
        if result.provenance:
            for parent in result.provenance:
                parent_hash = parent.get_content_hash()
                parent_id = self._get_by_hash(parent_hash)
                if parent_id:
                    cursor.execute(
                        "INSERT INTO provenance (result_id, parent_id) VALUES (?, ?)",
                        (result_id, parent_id)
                    )

        conn.commit()
        conn.close()

        self._log(f"Stored result with ID: {result_id}")

        # Check cache size and cleanup if needed
        if self.auto_cache:
            self._cleanup_cache()

        return result_id

    def get(self, result_id: str) -> Optional[MechIntResult]:
        """
        Retrieve a result by ID.

        Args:
            result_id: Unique identifier

        Returns:
            MechIntResult or None if not found
        """
        conn = sqlite3.connect(self.metadata_db)
        cursor = conn.cursor()

        cursor.execute("SELECT file_path, result_type FROM results WHERE result_id = ?", (result_id,))
        row = cursor.fetchone()
        conn.close()

        if not row:
            self._log(f"Result not found: {result_id}")
            return None

        file_path, result_type = row

        # Load appropriate class
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

        return result_class.load(file_path)

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
        Query results by metadata criteria.

        Args:
            method: Filter by method name
            result_type: Filter by result type
            tags: Filter by tags (all must match)
            start_time: Filter by timestamp >= start_time
            end_time: Filter by timestamp <= end_time
            limit: Maximum number of results

        Returns:
            List of result IDs matching criteria
        """
        conn = sqlite3.connect(self.metadata_db)
        cursor = conn.cursor()

        query = "SELECT DISTINCT r.result_id FROM results r"
        conditions = []
        params = []

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

        if tags:
            # All tags must match
            tag_conditions = " OR ".join(["t.tag = ?"] * len(tags))
            conditions.append(f"({tag_conditions})")
            params.extend(tags)
            query += f" GROUP BY r.result_id HAVING COUNT(DISTINCT t.tag) = {len(tags)}"

        if start_time:
            conditions.append("r.timestamp >= ?")
            params.append(start_time)

        if end_time:
            conditions.append("r.timestamp <= ?")
            params.append(end_time)

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

        similar = []
        for candidate_id in candidates:
            candidate = self.get(candidate_id)
            if candidate:
                comparison = result.compare(candidate)
                # Use mean of all comparison metrics as similarity score
                if comparison:
                    similarity = np.mean(list(comparison.values()))
                    if similarity >= threshold:
                        similar.append((candidate_id, float(similarity)))

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
        results = {}

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

            # Delete from database
            cursor.execute("DELETE FROM tags WHERE result_id = ?", (result_id,))
            cursor.execute("DELETE FROM provenance WHERE result_id = ? OR parent_id = ?",
                          (result_id, result_id))
            cursor.execute("DELETE FROM results WHERE result_id = ?", (result_id,))

            conn.commit()
            self._log(f"Deleted result: {result_id}")

        conn.close()

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

    def _get_by_hash(self, content_hash: str) -> Optional[str]:
        """Get result ID by content hash."""
        conn = sqlite3.connect(self.metadata_db)
        cursor = conn.cursor()
        cursor.execute("SELECT result_id FROM results WHERE content_hash = ?", (content_hash,))
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else None

    def _cleanup_cache(self):
        """Remove oldest results if cache exceeds size limit."""
        stats = self.get_stats()

        if stats['total_size_gb'] > self.max_cache_size_gb:
            # Calculate how much to delete
            to_delete_gb = stats['total_size_gb'] - self.max_cache_size_gb * 0.8  # Keep 80% of limit

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
        params = []

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
            min_effect_size: Minimum effect size
            limit: Maximum number of results

        Returns:
            List of result IDs
        """
        conn = sqlite3.connect(self.metadata_db)
        cursor = conn.cursor()

        query = "SELECT result_id FROM results WHERE result_type = 'InterventionResult'"
        params = []

        if intervention_type:
            query += " AND metadata_json LIKE ?"
            params.append(f'%"intervention_type": "{intervention_type}"%')

        # Note: For effect_size filtering, we'd need to parse metrics_json
        # This is a simplified version

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

        result_ids = []
        for result_id, metrics_json in rows:
            if near_critical:
                # Parse metrics to check branching parameter
                try:
                    metrics = json.loads(metrics_json)
                    branching = metrics.get('branching_parameter', 0)
                    if abs(branching - 1.0) < threshold:
                        result_ids.append(result_id)
                except:
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

        result_ids = []
        for result_id, metadata_json, metrics_json in rows:
            try:
                metadata = json.loads(metadata_json)
                metrics = json.loads(metrics_json)

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

            except:
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

        summary = {
            'total_results': stats['total_results'],
            'total_size_gb': stats['total_size_gb'],
            'by_method': stats['by_method'],
            'by_type': stats['by_type'],
        }

        # Add specialized counts
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
                metrics = json.loads(row[0])
                if abs(metrics.get('branching_parameter', 0) - 1.0) < 0.1:
                    near_critical_count += 1
            except:
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


__all__ = [
    'MechIntDatabase',
]
