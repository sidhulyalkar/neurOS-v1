"""Export astrocyte events to Parquet format."""

from pathlib import Path
import pandas as pd
from neuros_astro.metadata.schema import AstroEvent


def events_to_dataframe(events: list[AstroEvent]) -> pd.DataFrame:
    """
    Convert list of AstroEvent objects to pandas DataFrame.

    Args:
        events: List of AstroEvent objects

    Returns:
        DataFrame with event data
    """
    if len(events) == 0:
        # Return empty DataFrame with correct columns
        return pd.DataFrame(columns=[
            "event_id",
            "session_id",
            "region_id",
            "onset_frame",
            "offset_frame",
            "peak_frame",
            "duration_s",
            "peak_dff",
            "area_px",
            "centroid_y",
            "centroid_x",
            "propagation_speed",
            "direction_rad",
            "confidence",
        ])

    records = []

    for event in events:
        record = {
            "event_id": event.event_id,
            "session_id": event.session_id,
            "region_id": event.region_id,
            "onset_frame": event.onset_frame,
            "offset_frame": event.offset_frame,
            "peak_frame": event.peak_frame,
            "duration_s": event.duration_s,
            "peak_dff": event.peak_dff,
            "area_px": event.area_px,
            "centroid_y": event.centroid_yx[0] if event.centroid_yx else None,
            "centroid_x": event.centroid_yx[1] if event.centroid_yx else None,
            "propagation_speed": event.propagation_speed,
            "direction_rad": event.direction_rad,
            "confidence": event.confidence,
        }

        records.append(record)

    return pd.DataFrame(records)


def dataframe_to_events(df: pd.DataFrame) -> list[AstroEvent]:
    """
    Convert pandas DataFrame to list of AstroEvent objects.

    Args:
        df: DataFrame with event data

    Returns:
        List of AstroEvent objects
    """
    events = []

    for _, row in df.iterrows():
        # Reconstruct centroid
        if pd.notna(row.get("centroid_y")) and pd.notna(row.get("centroid_x")):
            centroid_yx = (float(row["centroid_y"]), float(row["centroid_x"]))
        else:
            centroid_yx = None

        event = AstroEvent(
            event_id=str(row["event_id"]),
            session_id=str(row["session_id"]),
            region_id=str(row["region_id"]) if pd.notna(row.get("region_id")) else None,
            onset_frame=int(row["onset_frame"]),
            offset_frame=int(row["offset_frame"]),
            peak_frame=int(row["peak_frame"]),
            duration_s=float(row["duration_s"]),
            peak_dff=float(row["peak_dff"]),
            area_px=float(row["area_px"]) if pd.notna(row.get("area_px")) else None,
            centroid_yx=centroid_yx,
            propagation_speed=float(row["propagation_speed"])
            if pd.notna(row.get("propagation_speed"))
            else None,
            direction_rad=float(row["direction_rad"]) if pd.notna(row.get("direction_rad")) else None,
            confidence=float(row["confidence"]) if pd.notna(row.get("confidence")) else 1.0,
        )

        events.append(event)

    return events


def save_events_parquet(events: list[AstroEvent], path: str | Path) -> None:
    """
    Save events to Parquet file.

    Args:
        events: List of AstroEvent objects
        path: Output file path
    """
    df = events_to_dataframe(events)
    df.to_parquet(path, index=False, engine="pyarrow")


def load_events_parquet(path: str | Path) -> list[AstroEvent]:
    """
    Load events from Parquet file.

    Args:
        path: Input file path

    Returns:
        List of AstroEvent objects
    """
    df = pd.read_parquet(path)
    return dataframe_to_events(df)
