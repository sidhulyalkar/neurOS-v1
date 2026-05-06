"""Main CLI application for neuros-astro."""

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
import numpy as np

from neuros_astro.metadata.dataset_scoring import scan_metadata_file, score_dataset_metadata
from neuros_astro.io.synthetic import generate_synthetic_astro_traces, generate_synthetic_astro_movie
from neuros_astro.events.event_detection import detect_events_from_traces, detect_candidate_events_from_movie
from neuros_astro.networks.functional_connectivity import build_event_coactivation_graph
from neuros_astro.tokenization.event_tokenizer import AstroEventTokenizer
from neuros_astro.export.to_parquet import save_events_parquet, load_events_parquet
from neuros_astro.export.to_neurofm import save_tokenized_sequence_npz, build_neurofm_manifest, save_neurofm_manifest

app = typer.Typer(help="neuros-astro: A glial signal processing layer for neural foundation models")
console = Console()


@app.command()
def scan(
    path: str = typer.Argument(..., help="Path to metadata file (JSON, NWB, or text)"),
    out: Optional[str] = typer.Option(None, help="Output JSON file path"),
):
    """
    Scan a dataset for astrocyte reanalysis potential.

    Supports JSON metadata files, NWB files, and plain text descriptions.
    """
    console.print(f"[bold blue]Scanning:[/bold blue] {path}")

    try:
        result = scan_metadata_file(path)

        # Print summary table
        table = Table(title="Dataset Triage Result")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Session ID", result.session_id)
        table.add_row("Astro Score", f"{result.astro_reanalysis_score:.2f}")
        table.add_row("Has Raw Movie", "✓" if result.has_raw_movie else "✗")
        table.add_row("Has Masks", "✓" if result.has_masks else "✗")
        table.add_row("Has Behavior", "✓" if result.has_behavior else "✗")
        table.add_row("Has Ephys", "✓" if result.has_ephys else "✗")
        table.add_row("Recommended Step", result.recommended_next_step)

        console.print(table)

        if result.matched_astro_terms:
            console.print(f"\n[bold]Matched astro terms:[/bold] {', '.join(result.matched_astro_terms)}")
        if result.matched_calcium_terms:
            console.print(f"[bold]Matched calcium terms:[/bold] {', '.join(result.matched_calcium_terms)}")
        if result.matched_modality_terms:
            console.print(f"[bold]Matched modality terms:[/bold] {', '.join(result.matched_modality_terms)}")

        if result.warnings:
            console.print("\n[bold yellow]Warnings:[/bold yellow]")
            for warning in result.warnings:
                console.print(f"  - {warning}")

        # Save to file if requested
        if out:
            out_path = Path(out)
            out_path.parent.mkdir(parents=True, exist_ok=True)

            with open(out_path, "w") as f:
                json.dump(result.model_dump(), f, indent=2)

            console.print(f"\n[bold green]Saved report to:[/bold green] {out}")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1)


@app.command()
def generate_synthetic(
    out_dir: str = typer.Option("./examples/data", help="Output directory"),
    frame_rate: float = typer.Option(10.0, help="Frame rate in Hz"),
    duration: float = typer.Option(60.0, help="Duration in seconds for traces"),
    n_regions: int = typer.Option(10, help="Number of regions for traces"),
    movie_duration: float = typer.Option(30.0, help="Duration in seconds for movie"),
    height: int = typer.Option(128, help="Movie height in pixels"),
    width: int = typer.Option(128, help="Movie width in pixels"),
    seed: int = typer.Option(42, help="Random seed"),
):
    """Generate synthetic astrocyte data for testing."""
    console.print("[bold blue]Generating synthetic astrocyte data...[/bold blue]")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Generate traces
    console.print(f"Generating traces: {n_regions} regions, {duration}s at {frame_rate}Hz")
    traces, trace_events = generate_synthetic_astro_traces(
        n_regions=n_regions,
        duration_s=duration,
        frame_rate_hz=frame_rate,
        seed=seed,
    )

    traces_path = out_path / "synthetic_traces.npy"
    np.save(traces_path, traces)
    console.print(f"  ✓ Saved traces to: {traces_path}")
    console.print(f"  ✓ Generated {len(trace_events)} ground truth events")

    # Save ground truth
    gt_path = out_path / "synthetic_traces_gt.json"
    with open(gt_path, "w") as f:
        json.dump(trace_events, f, indent=2)
    console.print(f"  ✓ Saved ground truth to: {gt_path}")

    # Generate movie
    console.print(f"\nGenerating movie: {height}x{width}, {movie_duration}s at {frame_rate}Hz")
    movie, movie_events = generate_synthetic_astro_movie(
        duration_s=movie_duration,
        frame_rate_hz=frame_rate,
        height=height,
        width=width,
        seed=seed,
    )

    movie_path = out_path / "synthetic_movie.npy"
    np.save(movie_path, movie)
    console.print(f"  ✓ Saved movie to: {movie_path}")
    console.print(f"  ✓ Generated {len(movie_events)} ground truth events")

    # Save ground truth
    gt_movie_path = out_path / "synthetic_movie_gt.json"
    with open(gt_movie_path, "w") as f:
        json.dump(movie_events, f, indent=2)
    console.print(f"  ✓ Saved ground truth to: {gt_movie_path}")

    console.print("\n[bold green]✓ Synthetic data generation complete![/bold green]")


@app.command()
def detect_trace_events(
    traces_path: str = typer.Argument(..., help="Path to traces NPY file"),
    frame_rate: float = typer.Option(..., help="Frame rate in Hz"),
    session_id: str = typer.Option(..., help="Session identifier"),
    out: str = typer.Option(..., help="Output Parquet file path"),
    z_threshold: float = typer.Option(2.0, help="Z-score threshold for detection"),
    min_duration: float = typer.Option(1.0, help="Minimum event duration in seconds"),
    merge_gap: float = typer.Option(0.5, help="Merge gap in seconds"),
):
    """Detect astrocyte events from calcium traces."""
    console.print(f"[bold blue]Detecting events from traces:[/bold blue] {traces_path}")

    # Load traces
    traces = np.load(traces_path)
    console.print(f"  Loaded traces: shape {traces.shape}")

    # Detect events
    events = detect_events_from_traces(
        traces=traces,
        frame_rate_hz=frame_rate,
        session_id=session_id,
        z_threshold=z_threshold,
        min_duration_s=min_duration,
        merge_gap_s=merge_gap,
    )

    console.print(f"  ✓ Detected {len(events)} events")

    # Save to Parquet
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_events_parquet(events, out_path)

    console.print(f"[bold green]✓ Saved events to:[/bold green] {out}")


@app.command()
def detect_movie_events(
    movie_path: str = typer.Argument(..., help="Path to movie NPY file"),
    frame_rate: float = typer.Option(..., help="Frame rate in Hz"),
    session_id: str = typer.Option(..., help="Session identifier"),
    out: str = typer.Option(..., help="Output Parquet file path"),
    z_threshold: float = typer.Option(3.0, help="Z-score threshold for detection"),
    min_area: int = typer.Option(10, help="Minimum area in pixels"),
    min_duration: float = typer.Option(0.5, help="Minimum event duration in seconds"),
):
    """Detect candidate spatiotemporal events from calcium movie."""
    console.print(f"[bold blue]Detecting events from movie:[/bold blue] {movie_path}")

    # Load movie
    movie = np.load(movie_path)
    console.print(f"  Loaded movie: shape {movie.shape}")

    # Detect events
    events = detect_candidate_events_from_movie(
        movie=movie,
        frame_rate_hz=frame_rate,
        session_id=session_id,
        z_threshold=z_threshold,
        min_area_px=min_area,
        min_duration_s=min_duration,
    )

    console.print(f"  ✓ Detected {len(events)} events")

    # Save to Parquet
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_events_parquet(events, out_path)

    console.print(f"[bold green]✓ Saved events to:[/bold green] {out}")


@app.command()
def build_network(
    events_path: str = typer.Argument(..., help="Path to events Parquet file"),
    frame_rate: float = typer.Option(..., help="Frame rate in Hz"),
    session_id: str = typer.Option(..., help="Session identifier"),
    out: str = typer.Option(..., help="Output JSON file path"),
    window_size: float = typer.Option(30.0, help="Window size in seconds"),
    stride: float = typer.Option(5.0, help="Stride in seconds"),
    min_edge_weight: float = typer.Option(0.1, help="Minimum edge weight"),
):
    """Build astrocyte coactivation networks from events."""
    console.print(f"[bold blue]Building networks from events:[/bold blue] {events_path}")

    # Load events
    events = load_events_parquet(events_path)
    console.print(f"  Loaded {len(events)} events")

    # Build graphs
    graphs = build_event_coactivation_graph(
        events=events,
        session_id=session_id,
        frame_rate_hz=frame_rate,
        window_size_s=window_size,
        stride_s=stride,
        min_edge_weight=min_edge_weight,
    )

    console.print(f"  ✓ Built {len(graphs)} graphs")

    # Save to JSON
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    graphs_data = [g.model_dump() for g in graphs]

    with open(out_path, "w") as f:
        json.dump(graphs_data, f, indent=2)

    console.print(f"[bold green]✓ Saved graphs to:[/bold green] {out}")


@app.command()
def tokenize_events(
    events_path: str = typer.Argument(..., help="Path to events Parquet file"),
    session_id: str = typer.Option(..., help="Session identifier"),
    out: str = typer.Option(..., help="Output NPZ file path"),
    normalize: bool = typer.Option(True, help="Normalize features"),
):
    """Tokenize astrocyte events for foundation models."""
    console.print(f"[bold blue]Tokenizing events:[/bold blue] {events_path}")

    # Load events
    events = load_events_parquet(events_path)
    console.print(f"  Loaded {len(events)} events")

    # Tokenize
    tokenizer = AstroEventTokenizer(normalize=normalize)
    tokens = tokenizer.tokenize(events, session_id=session_id)

    console.print(f"  ✓ Generated {len(tokens.tokens)} tokens")
    console.print(f"  ✓ Features: {', '.join(tokens.feature_names)}")

    # Save to NPZ
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_tokenized_sequence_npz(tokens, out_path)

    console.print(f"[bold green]✓ Saved tokens to:[/bold green] {out}")

    # Also save manifest
    manifest_path = out_path.parent / f"{out_path.stem}_manifest.json"
    manifest = build_neurofm_manifest(
        session_id=session_id,
        modalities={
            "astro": {
                "type": "event_tokens",
                "path": str(out_path.name),
                "sampling": "irregular",
                "timestamp_key": "timestamps_s",
            }
        },
    )
    save_neurofm_manifest(manifest, manifest_path)
    console.print(f"  ✓ Saved manifest to: {manifest_path}")


if __name__ == "__main__":
    app()
