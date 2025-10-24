"""
Audio Tokenizer for NeuroFMx

Processes audio streams (vocalizations, environmental sounds) for behavioral analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class AudioTokenizer(nn.Module):
    """
    Tokenize audio data using mel-spectrograms and temporal convolutions.

    Supports:
    - Vocalizations (speech, animal calls)
    - Environmental audio
    - Ultrasonic recordings
    """

    def __init__(
        self,
        d_model: int = 512,
        n_mels: int = 80,
        target_seq_len: int = 100,
        use_spectrogram: bool = True,
        sample_rate: int = 16000,
        hop_length: int = 160,
        n_fft: int = 400
    ):
        """
        Args:
            d_model: Output embedding dimension
            n_mels: Number of mel filterbanks
            target_seq_len: Target sequence length after tokenization
            use_spectrogram: Compute spectrogram internally
            sample_rate: Audio sampling rate
            hop_length: Hop length for STFT
            n_fft: FFT size
        """
        super().__init__()

        self.d_model = d_model
        self.n_mels = n_mels
        self.target_seq_len = target_seq_len
        self.use_spectrogram = use_spectrogram
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_fft = n_fft

        # Mel spectrogram (if using raw waveform input)
        if use_spectrogram:
            try:
                import torchaudio
                self.mel_transform = torchaudio.transforms.MelSpectrogram(
                    sample_rate=sample_rate,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    n_mels=n_mels
                )
            except ImportError:
                print("Warning: torchaudio not available, expecting pre-computed spectrograms")
                self.mel_transform = None

        # Frequency encoder: process mel bins
        self.freq_conv = nn.Sequential(
            nn.Conv1d(n_mels, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, d_model, kernel_size=3, padding=1),
        )

        # Temporal encoder: multi-scale temporal convolutions
        self.temporal_convs = nn.ModuleList([
            nn.Conv1d(d_model, d_model, kernel_size=k, padding=k//2)
            for k in [3, 5, 7, 9]
        ])

        self.temporal_proj = nn.Linear(d_model * 4, d_model)

        # Adaptive pooling to target length
        self.adaptive_pool = nn.AdaptiveAvgPool1d(target_seq_len)

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, target_seq_len, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Either:
               - Raw waveform: (batch, n_samples)
               - Mel spectrogram: (batch, n_mels, time) or (batch, time, n_mels)

        Returns:
            tokens: (batch, target_seq_len, d_model)
        """
        batch_size = x.shape[0]

        # Convert to mel spectrogram if needed
        if x.ndim == 2:  # Raw waveform
            if self.mel_transform is not None:
                x = self.mel_transform(x)  # (batch, n_mels, time)
            else:
                raise ValueError("Raw waveform provided but mel_transform not available")

        # Ensure shape is (batch, n_mels, time)
        if x.shape[1] != self.n_mels and x.shape[2] == self.n_mels:
            x = x.transpose(1, 2)

        # Frequency encoding
        freq_features = self.freq_conv(x)  # (batch, d_model, time)

        # Multi-scale temporal encoding
        temporal_features = []
        for conv in self.temporal_convs:
            feat = conv(freq_features)
            temporal_features.append(feat)

        # Concatenate multi-scale features
        combined = torch.cat(temporal_features, dim=1)  # (batch, d_model*4, time)
        combined = combined.transpose(1, 2)  # (batch, time, d_model*4)
        combined = self.temporal_proj(combined)  # (batch, time, d_model)

        # Adaptive pooling to target length
        combined = combined.transpose(1, 2)  # (batch, d_model, time)
        combined = self.adaptive_pool(combined)  # (batch, d_model, target_len)
        combined = combined.transpose(1, 2)  # (batch, target_len, d_model)

        # Add positional embedding
        combined = combined + self.pos_embed

        return combined


class VocalizationTokenizer(nn.Module):
    """
    Specialized tokenizer for vocalizations (speech, animal calls).

    Uses speech-specific features like MFCCs, pitch, formants.
    """

    def __init__(
        self,
        d_model: int = 512,
        n_mels: int = 80,
        n_mfcc: int = 13,
        target_seq_len: int = 100,
        extract_pitch: bool = True,
        sample_rate: int = 16000
    ):
        """
        Args:
            d_model: Output dimension
            n_mels: Number of mel bins
            n_mfcc: Number of MFCCs
            target_seq_len: Target sequence length
            extract_pitch: Extract pitch features
            sample_rate: Audio sample rate
        """
        super().__init__()

        self.d_model = d_model
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.target_seq_len = target_seq_len
        self.extract_pitch = extract_pitch
        self.sample_rate = sample_rate

        # Mel spectrogram
        try:
            import torchaudio
            self.mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_mels=n_mels
            )
            self.mfcc_transform = torchaudio.transforms.MFCC(
                sample_rate=sample_rate,
                n_mfcc=n_mfcc,
                melkwargs={'n_mels': n_mels}
            )
        except ImportError:
            print("Warning: torchaudio not available")
            self.mel_transform = None
            self.mfcc_transform = None

        # Feature encoders
        self.mel_encoder = nn.Conv1d(n_mels, d_model // 3, kernel_size=3, padding=1)
        self.mfcc_encoder = nn.Conv1d(n_mfcc, d_model // 3, kernel_size=3, padding=1)

        if extract_pitch:
            self.pitch_encoder = nn.Conv1d(1, d_model // 3, kernel_size=3, padding=1)

        # Temporal processing
        self.temporal_lstm = nn.LSTM(
            d_model,
            d_model,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        self.lstm_proj = nn.Linear(d_model * 2, d_model)

        # Adaptive pooling
        self.adaptive_pool = nn.AdaptiveAvgPool1d(target_seq_len)

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, target_seq_len, d_model))

    def extract_pitch(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract pitch using autocorrelation.

        Simplified pitch extraction for demonstration.
        """
        # Use FFT-based pitch detection
        # This is a placeholder - in production use librosa or crepe
        batch_size = waveform.shape[0]

        # Simple envelope as proxy for pitch energy
        pitch = torch.abs(waveform).unsqueeze(1)  # (batch, 1, n_samples)

        # Downsample
        pitch = F.adaptive_avg_pool1d(pitch, waveform.shape[1] // 160)

        return pitch

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: (batch, n_samples) raw audio

        Returns:
            tokens: (batch, target_seq_len, d_model)
        """
        batch_size = waveform.shape[0]

        features = []

        # Mel spectrogram
        if self.mel_transform is not None:
            mel = self.mel_transform(waveform)  # (batch, n_mels, time)
            mel_feat = self.mel_encoder(mel)  # (batch, d_model//3, time)
            features.append(mel_feat)

        # MFCCs
        if self.mfcc_transform is not None:
            mfcc = self.mfcc_transform(waveform)  # (batch, n_mfcc, time)
            mfcc_feat = self.mfcc_encoder(mfcc)  # (batch, d_model//3, time)
            features.append(mfcc_feat)

        # Pitch
        if self.extract_pitch:
            pitch = self.extract_pitch(waveform)  # (batch, 1, time)
            pitch_feat = self.pitch_encoder(pitch)  # (batch, d_model//3, time)
            features.append(pitch_feat)

        # Combine features
        combined = torch.cat(features, dim=1)  # (batch, d_model, time)
        combined = combined.transpose(1, 2)  # (batch, time, d_model)

        # Temporal modeling with LSTM
        combined, _ = self.temporal_lstm(combined)  # (batch, time, d_model*2)
        combined = self.lstm_proj(combined)  # (batch, time, d_model)

        # Adaptive pooling
        combined = combined.transpose(1, 2)  # (batch, d_model, time)
        combined = self.adaptive_pool(combined)  # (batch, d_model, target_len)
        combined = combined.transpose(1, 2)  # (batch, target_len, d_model)

        # Add positional embedding
        combined = combined + self.pos_embed

        return combined


class UltrasonicTokenizer(nn.Module):
    """
    Specialized tokenizer for ultrasonic vocalizations (USVs).

    Common in rodent behavioral studies.
    """

    def __init__(
        self,
        d_model: int = 512,
        n_mels: int = 128,
        target_seq_len: int = 100,
        sample_rate: int = 250000,  # High sampling rate for ultrasonics
        freq_range: tuple = (20000, 120000)  # USV frequency range
    ):
        super().__init__()

        self.d_model = d_model
        self.n_mels = n_mels
        self.target_seq_len = target_seq_len
        self.sample_rate = sample_rate
        self.freq_range = freq_range

        # High-frequency mel spectrogram
        try:
            import torchaudio
            self.mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_mels=n_mels,
                f_min=freq_range[0],
                f_max=freq_range[1]
            )
        except ImportError:
            self.mel_transform = None

        # USV-specific feature encoder
        self.freq_encoder = nn.Sequential(
            nn.Conv1d(n_mels, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, d_model, kernel_size=3, padding=1),
        )

        # Temporal encoder
        self.temporal_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead=8, batch_first=True),
            num_layers=2
        )

        # Adaptive pooling
        self.adaptive_pool = nn.AdaptiveAvgPool1d(target_seq_len)

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, target_seq_len, d_model))

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: (batch, n_samples) ultrasonic audio

        Returns:
            tokens: (batch, target_seq_len, d_model)
        """
        # Mel spectrogram
        if self.mel_transform is not None:
            mel = self.mel_transform(waveform)  # (batch, n_mels, time)
        else:
            # Assume pre-computed
            mel = waveform

        # Frequency encoding
        freq_feat = self.freq_encoder(mel)  # (batch, d_model, time)
        freq_feat = freq_feat.transpose(1, 2)  # (batch, time, d_model)

        # Temporal encoding
        temporal_feat = self.temporal_encoder(freq_feat)  # (batch, time, d_model)

        # Adaptive pooling
        temporal_feat = temporal_feat.transpose(1, 2)  # (batch, d_model, time)
        temporal_feat = self.adaptive_pool(temporal_feat)  # (batch, d_model, target_len)
        temporal_feat = temporal_feat.transpose(1, 2)  # (batch, target_len, d_model)

        # Add positional embedding
        temporal_feat = temporal_feat + self.pos_embed

        return temporal_feat
