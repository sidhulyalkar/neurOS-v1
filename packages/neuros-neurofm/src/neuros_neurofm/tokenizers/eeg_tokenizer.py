"""
EEG Tokenizer for NeuroFMx

Converts EEG signals into token embeddings for the foundation model.
Handles spatial (across channels) and temporal (over time) encoding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from neuros_neurofm.tokenizers.base_tokenizer import BaseTokenizer, TokenizedSequence


class EEGTokenizer(BaseTokenizer):
    """
    Tokenizer for EEG data.

    Applies spatial and temporal convolutions to multi-channel EEG,
    producing a sequence of embeddings.

    Args:
        n_channels: Number of EEG channels (e.g., 64 for standard 10-20 system)
        d_model: Output embedding dimension
        seq_len: Target sequence length
        sfreq: Sampling frequency (Hz)
        use_spectral: Whether to include spectral features
        dropout: Dropout probability
    """

    def __init__(
        self,
        n_channels: int = 64,
        d_model: int = 512,
        seq_len: int = 100,
        sfreq: float = 128.0,
        use_spectral: bool = True,
        dropout: float = 0.1
    ):
        super().__init__(d_model=d_model)

        self.n_channels = n_channels
        self.seq_len = seq_len
        self.sfreq = sfreq
        self.use_spectral = use_spectral
        self.default_sampling_rate = sfreq

        # Spatial encoding across channels
        # Treats channels as a spatial dimension
        self.spatial_conv = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Temporal encoding
        # Multiple kernel sizes to capture different temporal scales
        self.temporal_convs = nn.ModuleList([
            nn.Conv1d(64, d_model // 4, kernel_size=k, padding=k//2)
            for k in [3, 7, 15, 31]  # Multi-scale temporal
        ])

        self.temporal_norm = nn.BatchNorm1d(d_model)

        # Spectral features (optional)
        if use_spectral:
            self.spectral_encoder = SpectralEncoder(
                n_channels=n_channels,
                d_model=d_model,
                sfreq=sfreq
            )
            self.fusion_layer = nn.Linear(d_model * 2, d_model)

        # Learned positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        # Adaptive pooling to target sequence length
        self.adaptive_pool = nn.AdaptiveAvgPool1d(seq_len)

        # Final projection
        self.proj = nn.Linear(d_model, d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        t0: float = 0.0,
        return_sequence: bool = True
    ) -> Tuple[torch.Tensor, Optional[TokenizedSequence]]:
        """
        Forward pass.

        Args:
            x: EEG data (batch, time, channels)
            mask: Optional mask (batch, time)
            t0: Start time in seconds
            return_sequence: If True, also return TokenizedSequence

        Returns:
            tokens: (batch, seq_len, d_model)
            sequence: TokenizedSequence (if return_sequence=True, else None)
        """
        batch_size = x.shape[0]
        input_time_len = x.shape[1]

        # Transpose to (batch, channels, time) for conv
        x = x.transpose(1, 2)

        # Spatial encoding
        x = self.spatial_conv(x)  # (B, 64, T)

        # Multi-scale temporal encoding
        temporal_features = []
        for conv in self.temporal_convs:
            feat = F.relu(conv(x))
            temporal_features.append(feat)

        # Concatenate multi-scale features
        x = torch.cat(temporal_features, dim=1)  # (B, d_model, T)
        x = self.temporal_norm(x)

        # Spectral features (if enabled)
        if self.use_spectral:
            x_orig = x.transpose(1, 2)  # Back to (B, T, C) for spectral encoder
            spectral_feat = self.spectral_encoder(x_orig)  # (B, T, d_model)

            # Fuse temporal and spectral
            x = x.transpose(1, 2)  # (B, T, d_model)
            x = torch.cat([x, spectral_feat], dim=-1)  # (B, T, 2*d_model)
            x = self.fusion_layer(x)  # (B, T, d_model)
            x = x.transpose(1, 2)  # (B, d_model, T)

        # Adaptive pooling to target sequence length
        x = self.adaptive_pool(x)  # (B, d_model, seq_len)

        # Transpose back to (B, seq_len, d_model)
        x = x.transpose(1, 2)

        # Add positional encoding
        x = x + self.pos_encoding[:, :x.shape[1], :]

        # Final projection and dropout
        x = self.proj(x)
        x = self.dropout(x)

        # Apply mask if provided
        if mask is not None:
            # Downsample mask to match sequence length
            mask_pooled = F.adaptive_avg_pool1d(
                mask.unsqueeze(1).float(),
                self.seq_len
            ).squeeze(1) > 0.5
            x = x * mask_pooled.unsqueeze(-1)
        else:
            mask_pooled = None

        # Create TokenizedSequence if requested
        sequence = None
        if return_sequence:
            # Compute dt based on original input and downsampled output
            input_duration = input_time_len / self.sfreq
            dt = input_duration / self.seq_len

            sequence = self.create_sequence(
                tokens=x,
                t0=t0,
                dt=dt,
                mask=mask_pooled,
                metadata={
                    'modality': 'eeg',
                    'n_channels': self.n_channels,
                    'original_sfreq': self.sfreq,
                    'use_spectral': self.use_spectral
                }
            )

        return x, sequence


class SpectralEncoder(nn.Module):
    """
    Encodes spectral (frequency domain) features from EEG.

    Computes power in standard EEG bands:
    - Delta (0.5-4 Hz)
    - Theta (4-8 Hz)
    - Alpha (8-13 Hz)
    - Beta (13-30 Hz)
    - Gamma (30-50 Hz)
    """

    def __init__(self, n_channels: int, d_model: int, sfreq: float):
        super().__init__()

        self.n_channels = n_channels
        self.sfreq = sfreq

        # Frequency bands
        self.bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 50)
        }

        n_bands = len(self.bands)

        # MLP to project band powers to d_model
        self.band_proj = nn.Sequential(
            nn.Linear(n_channels * n_bands, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model)
        )

    def compute_band_power(self, x: torch.Tensor, low: float, high: float) -> torch.Tensor:
        """
        Compute band power using FFT.

        Args:
            x: (batch, time, channels)
            low: Low frequency (Hz)
            high: High frequency (Hz)

        Returns:
            power: (batch, channels) band power
        """
        # Simple bandpass approximation using FFT
        # In practice, use proper bandpass filter

        # FFT
        x_fft = torch.fft.rfft(x, dim=1)  # (B, freq, C)
        freqs = torch.fft.rfftfreq(x.shape[1], d=1/self.sfreq)

        # Select frequency range
        freq_mask = (freqs >= low) & (freqs <= high)

        # Power in band
        power = (x_fft.abs() ** 2)[:, freq_mask, :].mean(dim=1)  # (B, C)

        return power

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute spectral features.

        Args:
            x: (batch, time, channels)

        Returns:
            features: (batch, time, d_model)
        """
        batch_size, time_len, n_channels = x.shape

        # Window the data for time-resolved spectral features
        window_size = min(int(self.sfreq), time_len)  # 1 second windows
        hop_size = window_size // 2

        # Sliding window
        windows = []
        for start in range(0, time_len - window_size + 1, hop_size):
            window = x[:, start:start+window_size, :]
            windows.append(window)

        if len(windows) == 0:
            # Time series too short, use entire signal
            windows = [x]

        # Compute band powers for each window
        window_features = []
        for window in windows:
            band_powers = []
            for band_name, (low, high) in self.bands.items():
                power = self.compute_band_power(window, low, high)  # (B, C)
                band_powers.append(power)

            # Concatenate all bands: (B, C * n_bands)
            band_features = torch.cat(band_powers, dim=-1)

            # Project to d_model
            feat = self.band_proj(band_features)  # (B, d_model)
            window_features.append(feat)

        # Stack windows
        window_features = torch.stack(window_features, dim=1)  # (B, n_windows, d_model)

        # Upsample to match time dimension
        window_features = F.interpolate(
            window_features.transpose(1, 2),
            size=time_len,
            mode='linear',
            align_corners=False
        ).transpose(1, 2)

        return window_features


# Example usage
if __name__ == '__main__':
    from neuros_neurofm.tokenizers.temporal_alignment import TemporalAligner, InterpolationMethod

    # Test tokenizer
    batch_size = 4
    time_len = 256  # 2 seconds at 128 Hz
    n_channels = 64

    tokenizer = EEGTokenizer(
        n_channels=n_channels,
        d_model=512,
        seq_len=100,
        sfreq=128.0,
        use_spectral=True
    )

    # Dummy input
    eeg_data = torch.randn(batch_size, time_len, n_channels)

    # Forward pass (backward compatible)
    tokens, sequence = tokenizer(eeg_data, t0=0.0, return_sequence=True)

    print(f"Input shape: {eeg_data.shape}")
    print(f"Output shape: {tokens.shape}")  # Should be (4, 100, 512)
    print(f"Tokenizer parameters: {sum(p.numel() for p in tokenizer.parameters()):,}")

    if sequence is not None:
        print(f"\nTokenizedSequence: {sequence}")
        print(f"Duration: {sequence.duration:.3f}s")
        print(f"Sampling rate: {sequence.sampling_rate:.1f}Hz")

    # Example: Multi-modal alignment with video
    print("\n=== Multi-modal Alignment Example ===")

    # Create a simulated video sequence (30 fps)
    video_tokens = torch.randn(batch_size, 60, 512)  # 2 seconds at 30fps
    video_seq = TokenizedSequence(
        tokens=video_tokens,
        t0=0.0,
        dt=1/30.0,
        mask=torch.ones(batch_size, 60, dtype=torch.bool),
        metadata={'modality': 'video', 'fps': 30}
    )

    # Align EEG and video to common 50 Hz grid
    aligner = TemporalAligner()
    aligned_eeg, aligned_video = aligner.align_to_grid(
        sequences=[sequence, video_seq],
        target_dt=0.02,  # 50 Hz
        method=InterpolationMethod.LINEAR
    )

    print(f"Aligned EEG: {aligned_eeg}")
    print(f"Aligned video: {aligned_video}")

    # Validate alignment
    validation = aligner.validate_alignment([aligned_eeg, aligned_video])
    print(f"Alignment valid: {validation['valid']}")

    # Create sliding windows
    windows = aligner.create_windows(
        sequences=[aligned_eeg, aligned_video],
        window_size=1.0,  # 1 second windows
        hop_size=0.5,     # 50% overlap
        align_first=False  # Already aligned
    )
    print(f"Created {len(windows)} windows")
