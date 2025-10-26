"""
Temporal Alignment Demo for NeuroFMX

This example demonstrates how to use the temporal alignment utilities
to synchronize multi-modal neural recordings with different sampling rates.

Scenarios covered:
1. Basic alignment of EEG and video
2. Spike train alignment with continuous signals
3. Creating sliding windows for training
4. Multi-modal fusion
5. Handling missing data and jitter
"""

import torch
import numpy as np

from neuros_neurofm.tokenizers import (
    EEGTokenizer,
    SpikeTokenizer,
    TokenizedSequence,
    TemporalAligner,
    InterpolationMethod,
    resample_to_rate,
    align_and_concatenate,
)


def demo_basic_alignment():
    """Demo 1: Basic alignment of EEG and video."""
    print("=" * 70)
    print("DEMO 1: Basic Multi-Modal Alignment (EEG + Video)")
    print("=" * 70)

    # Simulate EEG data (128 Hz, 64 channels, 2 seconds)
    eeg_data = torch.randn(4, 256, 64)
    eeg_tokenizer = EEGTokenizer(n_channels=64, d_model=512, seq_len=100, sfreq=128.0)

    # Tokenize EEG
    _, eeg_seq = eeg_tokenizer(eeg_data, t0=0.0, return_sequence=True)
    print(f"\nEEG sequence: {eeg_seq}")

    # Simulate video data (30 fps, 2 seconds)
    video_tokens = torch.randn(4, 60, 512)
    video_seq = TokenizedSequence(
        tokens=video_tokens,
        t0=0.0,
        dt=1/30.0,
        mask=torch.ones(4, 60, dtype=torch.bool),
        metadata={'modality': 'video', 'fps': 30}
    )
    print(f"Video sequence: {video_seq}")

    # Align to common 50 Hz grid
    aligner = TemporalAligner()
    aligned_eeg, aligned_video = aligner.align_to_grid(
        sequences=[eeg_seq, video_seq],
        target_dt=0.02,  # 50 Hz
        method=InterpolationMethod.LINEAR
    )

    print(f"\nAligned EEG: {aligned_eeg}")
    print(f"Aligned video: {aligned_video}")

    # Validate alignment
    validation = aligner.validate_alignment([aligned_eeg, aligned_video])
    print(f"\nAlignment valid: {validation['valid']}")
    print(f"Common time range: {validation['checks']['time_overlap']['common_range']}")
    print(f"Duration: {validation['checks']['time_overlap']['duration']:.3f}s")


def demo_spike_alignment():
    """Demo 2: Aligning spike trains with continuous signals."""
    print("\n" + "=" * 70)
    print("DEMO 2: Spike Train Alignment with LFP")
    print("=" * 70)

    # Simulate spike data (96 units, 1000 spikes)
    spike_times = torch.rand(4, 1000) * 2000  # 2 seconds in ms
    spike_units = torch.randint(0, 96, (4, 1000))

    spike_tokenizer = SpikeTokenizer(n_units=96, d_model=512, bin_size_ms=1.0)
    _, _, spike_seq = spike_tokenizer(
        spike_times, spike_units, t0=0.0, return_sequence=True
    )
    print(f"\nSpike sequence: {spike_seq}")

    # Simulate LFP data (1000 Hz, 2 seconds)
    lfp_tokens = torch.randn(4, 200, 512)  # Downsampled to 100 tokens
    lfp_seq = TokenizedSequence(
        tokens=lfp_tokens,
        t0=0.0,
        dt=0.01,  # 100 Hz effective rate
        mask=torch.ones(4, 200, dtype=torch.bool),
        metadata={'modality': 'lfp', 'sampling_rate': 1000.0}
    )
    print(f"LFP sequence: {lfp_seq}")

    # Align both to 100 Hz
    aligner = TemporalAligner()
    aligned_spikes, aligned_lfp = aligner.align_to_grid(
        sequences=[spike_seq, lfp_seq],
        target_dt=0.01,
        method=InterpolationMethod.LINEAR
    )

    print(f"\nAligned spikes: {aligned_spikes}")
    print(f"Aligned LFP: {aligned_lfp}")


def demo_sliding_windows():
    """Demo 3: Creating sliding windows for training."""
    print("\n" + "=" * 70)
    print("DEMO 3: Sliding Windows for Training")
    print("=" * 70)

    # Create two aligned sequences
    seq1 = TokenizedSequence(
        tokens=torch.randn(2, 500, 512),  # 5 seconds at 100 Hz
        t0=0.0,
        dt=0.01,
        mask=torch.ones(2, 500, dtype=torch.bool),
        metadata={'modality': 'eeg'}
    )

    seq2 = TokenizedSequence(
        tokens=torch.randn(2, 500, 512),
        t0=0.0,
        dt=0.01,
        mask=torch.ones(2, 500, dtype=torch.bool),
        metadata={'modality': 'video'}
    )

    # Create 1-second windows with 50% overlap
    aligner = TemporalAligner()
    windows = aligner.create_windows(
        sequences=[seq1, seq2],
        window_size=1.0,  # 1 second
        hop_size=0.5,     # 0.5 second stride (50% overlap)
        align_first=False
    )

    print(f"\nCreated {len(windows)} windows")
    print(f"Window 0 - EEG: {windows[0][0]}")
    print(f"Window 0 - Video: {windows[0][1]}")

    # Demonstrate batch processing
    print("\nBatch processing windows:")
    for i, (eeg_win, video_win) in enumerate(windows[:3]):
        print(f"  Window {i}: EEG={eeg_win.seq_len} tokens, "
              f"Video={video_win.seq_len} tokens, "
              f"time=[{eeg_win.t0:.2f}, {eeg_win.end_time:.2f}]s")


def demo_multimodal_fusion():
    """Demo 4: Multi-modal fusion by concatenating embeddings."""
    print("\n" + "=" * 70)
    print("DEMO 4: Multi-Modal Fusion")
    print("=" * 70)

    # Create sequences from different modalities with different rates
    eeg_seq = TokenizedSequence(
        tokens=torch.randn(2, 200, 256),  # 256-dim embeddings
        t0=0.0,
        dt=0.01,  # 100 Hz
        mask=torch.ones(2, 200, dtype=torch.bool),
        metadata={'modality': 'eeg'}
    )

    video_seq = TokenizedSequence(
        tokens=torch.randn(2, 60, 384),   # 384-dim embeddings
        t0=0.0,
        dt=1/30.0,  # 30 fps
        mask=torch.ones(2, 60, dtype=torch.bool),
        metadata={'modality': 'video'}
    )

    audio_seq = TokenizedSequence(
        tokens=torch.randn(2, 400, 128),  # 128-dim embeddings
        t0=0.0,
        dt=0.005,  # 200 Hz
        mask=torch.ones(2, 400, dtype=torch.bool),
        metadata={'modality': 'audio'}
    )

    print(f"\nOriginal sequences:")
    print(f"  EEG:   {eeg_seq.sampling_rate:.0f} Hz, d_model={eeg_seq.d_model}")
    print(f"  Video: {video_seq.sampling_rate:.0f} Hz, d_model={video_seq.d_model}")
    print(f"  Audio: {audio_seq.sampling_rate:.0f} Hz, d_model={audio_seq.d_model}")

    # Align and concatenate into unified representation
    fused = align_and_concatenate([eeg_seq, video_seq, audio_seq])

    print(f"\nFused sequence: {fused}")
    print(f"  Total embedding dimension: {fused.d_model}")
    print(f"  Component modalities: {fused.metadata['component_modalities']}")


def demo_missing_data_handling():
    """Demo 5: Handling missing data and jitter."""
    print("\n" + "=" * 70)
    print("DEMO 5: Missing Data and Jitter Correction")
    print("=" * 70)

    # Create sequence with missing data
    tokens = torch.randn(2, 100, 512)
    mask = torch.ones(2, 100, dtype=torch.bool)

    # Simulate missing data (e.g., artifacts in recording)
    mask[:, 30:40] = False  # 10% missing in middle
    mask[:, 70:75] = False  # 5% missing later

    seq = TokenizedSequence(
        tokens=tokens,
        t0=0.0,
        dt=0.01,
        mask=mask,
        metadata={'modality': 'eeg'}
    )

    print(f"\nOriginal sequence:")
    print(f"  Total tokens: {seq.seq_len}")
    print(f"  Valid tokens (batch 0): {seq.valid_tokens[0].item()}")
    print(f"  Missing tokens: {seq.seq_len - seq.valid_tokens[0].item()}")

    # Impute missing data
    aligner = TemporalAligner()
    imputed = aligner.impute_missing(seq, method=InterpolationMethod.LINEAR)

    print(f"\nImputed sequence:")
    print(f"  Valid tokens (batch 0): {imputed.valid_tokens[0].item()}")
    print(f"  All data restored: {imputed.mask.all().item()}")

    # Demonstrate jitter correction
    print("\n--- Jitter Correction ---")

    # Create sequence with timing jitter
    jittered_seq = TokenizedSequence(
        tokens=torch.randn(1, 50, 512),
        t0=0.0,
        dt=0.01,  # Nominal 100 Hz
        mask=torch.ones(1, 50, dtype=torch.bool),
        metadata={}
    )

    corrected = aligner.correct_jitter(jittered_seq, max_jitter=0.002)

    print(f"Original dt: {jittered_seq.dt:.6f}s")
    print(f"Corrected dt: {corrected.dt:.6f}s")


def demo_interpolation_methods():
    """Demo 6: Comparing interpolation methods."""
    print("\n" + "=" * 70)
    print("DEMO 6: Interpolation Method Comparison")
    print("=" * 70)

    # Create a simple sequence with known pattern
    t = torch.linspace(0, 1, 10)
    tokens = torch.sin(2 * np.pi * t).view(1, 10, 1)
    seq = TokenizedSequence(
        tokens=tokens,
        t0=0.0,
        dt=0.1,
        mask=torch.ones(1, 10, dtype=torch.bool),
        metadata={}
    )

    # Upsample to 50 points
    new_times = torch.linspace(0.0, 0.9, 50)

    aligner = TemporalAligner()

    methods = [
        InterpolationMethod.NEAREST,
        InterpolationMethod.LINEAR,
        InterpolationMethod.CUBIC,
        InterpolationMethod.CAUSAL
    ]

    print(f"\nOriginal sequence: 10 samples")
    print(f"Upsampled to: 50 samples\n")

    for method in methods:
        resampled = aligner.interpolate_sequence(seq, new_times, method)
        print(f"{method.value:10s}: seq_len={resampled.seq_len}, "
              f"dt={resampled.dt:.4f}s, "
              f"sampling_rate={resampled.sampling_rate:.1f}Hz")


def demo_synchronization_detection():
    """Demo 7: Detecting synchronization points."""
    print("\n" + "=" * 70)
    print("DEMO 7: Synchronization Point Detection")
    print("=" * 70)

    # Create sequences with synchronization events (sharp transitions)
    seq1 = TokenizedSequence(
        tokens=torch.randn(1, 200, 512),
        t0=0.0,
        dt=0.01,
        mask=torch.ones(1, 200, dtype=torch.bool),
        metadata={}
    )

    seq2 = TokenizedSequence(
        tokens=torch.randn(1, 200, 512),
        t0=0.0,
        dt=0.01,
        mask=torch.ones(1, 200, dtype=torch.bool),
        metadata={}
    )

    # Add artificial sync events at same time points
    sync_indices = [50, 100, 150]
    for idx in sync_indices:
        seq1.tokens[:, idx, :] *= 5  # Sharp increase
        seq2.tokens[:, idx, :] *= 5

    aligner = TemporalAligner()
    sync_points = aligner.detect_sync_points(
        sequences=[seq1, seq2],
        similarity_threshold=0.7
    )

    print(f"\nDetected {len(sync_points)} synchronization points:")
    for i, t in enumerate(sync_points[:5]):  # Show first 5
        print(f"  Sync {i+1}: t={t:.3f}s")


def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print("TEMPORAL ALIGNMENT UTILITIES - COMPREHENSIVE DEMO")
    print("=" * 70)

    demo_basic_alignment()
    demo_spike_alignment()
    demo_sliding_windows()
    demo_multimodal_fusion()
    demo_missing_data_handling()
    demo_interpolation_methods()
    demo_synchronization_detection()

    print("\n" + "=" * 70)
    print("All demos completed successfully!")
    print("=" * 70)


if __name__ == '__main__':
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    main()
