"""Synthetic astrocyte data generation for testing and validation."""

import numpy as np
from typing import Any


def generate_synthetic_astro_traces(
    n_regions: int = 10,
    duration_s: float = 60.0,
    frame_rate_hz: float = 10.0,
    n_events_per_region: int = 5,
    event_rise_time_s: tuple[float, float] = (1.0, 3.0),
    event_decay_time_s: tuple[float, float] = (3.0, 10.0),
    event_amplitude: tuple[float, float] = (0.1, 0.5),
    noise_std: float = 0.05,
    coactivation_prob: float = 0.2,
    seed: int | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """
    Generate synthetic astrocyte calcium traces with slow events.

    Astrocyte calcium events are typically slower than neuronal somatic events,
    with longer rise and decay times.

    Args:
        n_regions: Number of astrocyte regions
        duration_s: Total duration in seconds
        frame_rate_hz: Imaging frame rate
        n_events_per_region: Average number of events per region
        event_rise_time_s: Range of rise times (min, max)
        event_decay_time_s: Range of decay times (min, max)
        event_amplitude: Range of peak dF/F amplitudes (min, max)
        noise_std: Standard deviation of Gaussian noise
        coactivation_prob: Probability of cross-region coactivation
        seed: Random seed for reproducibility

    Returns:
        traces: Array of shape [n_regions, n_frames]
        ground_truth_events: List of ground truth event dictionaries
    """
    if seed is not None:
        np.random.seed(seed)

    n_frames = int(duration_s * frame_rate_hz)
    dt = 1.0 / frame_rate_hz

    # Initialize traces with baseline
    traces = np.zeros((n_regions, n_frames), dtype=np.float32)

    # Add baseline drift (very slow)
    for i in range(n_regions):
        baseline_freq = np.random.uniform(0.01, 0.05)  # Very slow drift
        baseline_amp = np.random.uniform(0.02, 0.08)
        time_points = np.arange(n_frames) * dt
        traces[i, :] += baseline_amp * np.sin(2 * np.pi * baseline_freq * time_points)

    # Generate events
    ground_truth_events = []

    for region_idx in range(n_regions):
        # Determine number of events for this region
        n_events = np.random.poisson(n_events_per_region)

        for event_idx in range(n_events):
            # Random event onset
            onset_frame = np.random.randint(0, max(1, n_frames - 100))

            # Event parameters
            rise_time = np.random.uniform(*event_rise_time_s)
            decay_time = np.random.uniform(*event_decay_time_s)
            amplitude = np.random.uniform(*event_amplitude)

            # Generate event waveform (alpha function for calcium)
            rise_frames = int(rise_time * frame_rate_hz)
            decay_frames = int(decay_time * frame_rate_hz)
            total_frames = rise_frames + decay_frames

            if onset_frame + total_frames >= n_frames:
                continue

            # Alpha function: t * exp(-t/tau)
            t_rise = np.arange(rise_frames) / frame_rate_hz
            t_decay = np.arange(decay_frames) / frame_rate_hz

            # Rise phase
            rise_waveform = amplitude * (t_rise / rise_time) * np.exp(1 - t_rise / rise_time)

            # Decay phase
            decay_waveform = amplitude * np.exp(-t_decay / decay_time)

            waveform = np.concatenate([rise_waveform, decay_waveform])

            # Add to trace
            end_frame = onset_frame + len(waveform)
            traces[region_idx, onset_frame:end_frame] += waveform

            # Find peak frame
            peak_frame = onset_frame + np.argmax(waveform)
            offset_frame = onset_frame + len(waveform) - 1

            # Store ground truth
            ground_truth_events.append({
                "region_id": f"roi_{region_idx:03d}",
                "onset_frame": int(onset_frame),
                "offset_frame": int(offset_frame),
                "peak_frame": int(peak_frame),
                "duration_s": float(len(waveform) * dt),
                "peak_dff": float(amplitude),
            })

            # Coactivation: trigger events in other regions
            if np.random.rand() < coactivation_prob:
                coactive_regions = np.random.choice(
                    [r for r in range(n_regions) if r != region_idx],
                    size=min(3, n_regions - 1),
                    replace=False,
                )

                for coactive_idx in coactive_regions:
                    # Slightly delayed and attenuated
                    delay_frames = np.random.randint(1, 10)
                    coactive_onset = onset_frame + delay_frames
                    coactive_amp = amplitude * np.random.uniform(0.5, 0.9)

                    if coactive_onset + total_frames < n_frames:
                        # Same waveform shape, different amplitude
                        coactive_waveform = waveform * (coactive_amp / amplitude)
                        coactive_end = coactive_onset + len(coactive_waveform)
                        traces[coactive_idx, coactive_onset:coactive_end] += coactive_waveform

                        ground_truth_events.append({
                            "region_id": f"roi_{coactive_idx:03d}",
                            "onset_frame": int(coactive_onset),
                            "offset_frame": int(coactive_onset + len(coactive_waveform) - 1),
                            "peak_frame": int(coactive_onset + np.argmax(coactive_waveform)),
                            "duration_s": float(len(coactive_waveform) * dt),
                            "peak_dff": float(coactive_amp),
                        })

    # Add noise
    noise = np.random.normal(0, noise_std, traces.shape)
    traces += noise

    return traces.astype(np.float32), ground_truth_events


def generate_synthetic_astro_movie(
    duration_s: float = 30.0,
    frame_rate_hz: float = 10.0,
    height: int = 128,
    width: int = 128,
    n_events: int = 10,
    event_size_px: tuple[int, int] = (10, 50),
    event_duration_s: tuple[float, float] = (2.0, 8.0),
    event_amplitude: tuple[float, float] = (0.2, 0.6),
    expansion_rate: tuple[float, float] = (2.0, 8.0),
    propagation_prob: float = 0.3,
    noise_std: float = 0.05,
    seed: int | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """
    Generate synthetic astrocyte calcium movie with spatiotemporal events.

    Astrocyte events can expand, propagate, and have complex spatial structure.

    Args:
        duration_s: Total duration in seconds
        frame_rate_hz: Frame rate
        height: Movie height in pixels
        width: Movie width in pixels
        n_events: Number of events to generate
        event_size_px: Range of event sizes (min, max) in pixels
        event_duration_s: Range of event durations (min, max)
        event_amplitude: Range of peak amplitudes (min, max)
        expansion_rate: Range of expansion rates in pixels/second
        propagation_prob: Probability of event propagation
        noise_std: Standard deviation of noise
        seed: Random seed

    Returns:
        movie: Array of shape [n_frames, height, width]
        ground_truth_events: List of ground truth event dictionaries
    """
    if seed is not None:
        np.random.seed(seed)

    n_frames = int(duration_s * frame_rate_hz)
    dt = 1.0 / frame_rate_hz

    # Initialize movie
    movie = np.zeros((n_frames, height, width), dtype=np.float32)

    # Create coordinate grids
    y_grid, x_grid = np.ogrid[:height, :width]

    ground_truth_events = []

    for event_idx in range(n_events):
        # Random event onset
        onset_frame = np.random.randint(0, max(1, n_frames - 50))

        # Event parameters
        duration = np.random.uniform(*event_duration_s)
        duration_frames = int(duration * frame_rate_hz)
        amplitude = np.random.uniform(*event_amplitude)
        initial_size = np.random.uniform(*event_size_px)
        expansion = np.random.uniform(*expansion_rate)

        # Random starting location (avoid edges)
        margin = min(int(max(event_size_px) * 2), min(height, width) // 4)
        margin = max(margin, 5)  # Minimum margin
        center_y = np.random.randint(margin, max(margin + 1, height - margin))
        center_x = np.random.randint(margin, max(margin + 1, width - margin))

        # Generate temporal profile (alpha function)
        t = np.arange(duration_frames) * dt
        tau_rise = duration * 0.2
        tau_decay = duration * 0.5
        temporal_profile = amplitude * (t / tau_rise) * np.exp(1 - t / tau_rise) * np.exp(-t / tau_decay)

        if onset_frame + duration_frames >= n_frames:
            continue

        # Create event frames
        for frame_offset in range(duration_frames):
            frame_idx = onset_frame + frame_offset

            # Expanding Gaussian blob
            size_at_t = initial_size + expansion * frame_offset * dt
            sigma = size_at_t / 2.355  # Convert FWHM to sigma

            # Gaussian spatial profile
            distances_sq = (y_grid - center_y) ** 2 + (x_grid - center_x) ** 2
            spatial_profile = np.exp(-distances_sq / (2 * sigma ** 2))

            # Add to movie
            movie[frame_idx, :, :] += temporal_profile[frame_offset] * spatial_profile

        # Find peak frame
        peak_frame = onset_frame + np.argmax(temporal_profile)
        offset_frame = onset_frame + duration_frames - 1

        # Calculate spatial extent at peak
        peak_size = initial_size + expansion * (peak_frame - onset_frame) * dt
        area_px = np.pi * (peak_size / 2) ** 2

        ground_truth_events.append({
            "event_id": f"evt_{event_idx:03d}",
            "onset_frame": int(onset_frame),
            "offset_frame": int(offset_frame),
            "peak_frame": int(peak_frame),
            "duration_s": float(duration),
            "peak_dff": float(amplitude),
            "area_px": float(area_px),
            "centroid_yx": (float(center_y), float(center_x)),
        })

        # Propagation: create a secondary event nearby
        if np.random.rand() < propagation_prob:
            # Propagate in random direction
            angle = np.random.uniform(0, 2 * np.pi)
            distance = np.random.uniform(20, 60)
            prop_center_y = int(center_y + distance * np.sin(angle))
            prop_center_x = int(center_x + distance * np.cos(angle))

            # Check bounds
            if margin <= prop_center_y < height - margin and margin <= prop_center_x < width - margin:
                prop_onset = onset_frame + np.random.randint(5, 15)
                prop_amplitude = amplitude * np.random.uniform(0.5, 0.8)

                if prop_onset + duration_frames < n_frames:
                    prop_temporal = prop_amplitude * (t / tau_rise) * np.exp(1 - t / tau_rise) * np.exp(
                        -t / tau_decay
                    )

                    for frame_offset in range(duration_frames):
                        frame_idx = prop_onset + frame_offset
                        if frame_idx >= n_frames:
                            break

                        size_at_t = initial_size + expansion * frame_offset * dt
                        sigma = size_at_t / 2.355

                        distances_sq = (y_grid - prop_center_y) ** 2 + (x_grid - prop_center_x) ** 2
                        spatial_profile = np.exp(-distances_sq / (2 * sigma ** 2))

                        movie[frame_idx, :, :] += prop_temporal[frame_offset] * spatial_profile

                    ground_truth_events.append({
                        "event_id": f"evt_{event_idx:03d}_prop",
                        "onset_frame": int(prop_onset),
                        "offset_frame": int(prop_onset + duration_frames - 1),
                        "peak_frame": int(prop_onset + np.argmax(prop_temporal)),
                        "duration_s": float(duration),
                        "peak_dff": float(prop_amplitude),
                        "area_px": float(area_px),
                        "centroid_yx": (float(prop_center_y), float(prop_center_x)),
                        "direction_rad": float(angle),
                    })

    # Add noise
    noise = np.random.normal(0, noise_std, movie.shape)
    movie += noise

    # Ensure non-negative
    movie = np.maximum(movie, 0)

    return movie.astype(np.float32), ground_truth_events
