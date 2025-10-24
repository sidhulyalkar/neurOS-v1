"""
Video Tokenizer for NeuroFMx

Processes behavioral video streams using 3D CNNs and spatiotemporal transformers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class VideoTokenizer(nn.Module):
    """
    Tokenize video data for behavioral analysis.

    Uses 3D convolutions to extract spatiotemporal features from video,
    then projects to model dimension.

    Supports:
    - Behavioral tracking videos
    - Naturalistic scene videos
    - Multi-view recordings
    """

    def __init__(
        self,
        d_model: int = 512,
        img_size: tuple = (224, 224),
        patch_size: int = 16,
        n_frames: int = 16,
        temporal_stride: int = 4,
        in_channels: int = 3,
        use_pretrained: bool = False
    ):
        """
        Args:
            d_model: Output embedding dimension
            img_size: Spatial size of input frames (H, W)
            patch_size: Spatial patch size for tokenization
            n_frames: Number of frames per clip
            temporal_stride: Temporal stride for 3D conv
            in_channels: Number of input channels (3 for RGB)
            use_pretrained: Use pretrained weights (e.g., from ImageNet/Kinetics)
        """
        super().__init__()

        self.d_model = d_model
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_frames = n_frames
        self.temporal_stride = temporal_stride
        self.in_channels = in_channels

        # Number of spatial patches
        self.n_patches_h = img_size[0] // patch_size
        self.n_patches_w = img_size[1] // patch_size
        self.n_patches_spatial = self.n_patches_h * self.n_patches_w

        # Number of temporal patches
        self.n_patches_temporal = n_frames // temporal_stride

        # 3D Stem: Extract spatiotemporal features
        self.stem = nn.Sequential(
            nn.Conv3d(
                in_channels,
                64,
                kernel_size=(3, 7, 7),
                stride=(1, 2, 2),
                padding=(1, 3, 3),
                bias=False
            ),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )

        # 3D ResNet-style blocks
        self.res_blocks = nn.ModuleList([
            self._make_3d_block(64, 128, stride=(1, 2, 2)),
            self._make_3d_block(128, 256, stride=(2, 2, 2)),
            self._make_3d_block(256, 512, stride=(2, 2, 2)),
        ])

        # Projection to d_model
        self.proj = nn.Linear(512, d_model)

        # Positional embedding
        # After all conv: T' = n_frames // 8, H' = img_size[0] // 32, W' = img_size[1] // 32
        self.temporal_embed = nn.Parameter(torch.randn(1, n_frames // 8, 1, 1, d_model))
        self.spatial_embed = nn.Parameter(torch.randn(1, 1, img_size[0] // 32, img_size[1] // 32, d_model))

    def _make_3d_block(self, in_channels: int, out_channels: int, stride: tuple):
        """Create a 3D residual block."""
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Video tensor of shape (batch, channels, frames, height, width)
               or (batch, frames, channels, height, width)

        Returns:
            tokens: (batch, n_tokens, d_model)
        """
        batch_size = x.shape[0]

        # Ensure shape is (batch, channels, frames, height, width)
        if x.shape[2] == self.in_channels:
            # Input is (batch, frames, channels, height, width)
            x = x.permute(0, 2, 1, 3, 4)

        # Apply stem
        x = self.stem(x)  # (B, 64, T, H, W)

        # Apply residual blocks
        for block in self.res_blocks:
            x = block(x)

        # x shape: (B, 512, T', H', W')
        # Project to d_model
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1).contiguous()  # (B, T', H', W', C)
        x = self.proj(x)  # (B, T', H', W', d_model)

        # Add positional embeddings
        x = x + self.temporal_embed[:, :T, :, :, :] + self.spatial_embed[:, :, :H, :W, :]

        # Flatten spatial dimensions
        x = x.reshape(B, T * H * W, self.d_model)

        return x


class VideoAudioTokenizer(nn.Module):
    """
    Joint video-audio tokenizer for multimodal behavioral data.

    Processes synchronized video and audio streams.
    """

    def __init__(
        self,
        d_model: int = 512,
        # Video params
        img_size: tuple = (224, 224),
        patch_size: int = 16,
        n_frames: int = 16,
        video_channels: int = 3,
        # Audio params
        n_mels: int = 80,
        audio_len: int = 1000,
        # Fusion
        fusion_type: str = 'concat'  # 'concat', 'add', 'cross_attention'
    ):
        super().__init__()

        self.d_model = d_model
        self.fusion_type = fusion_type

        # Video encoder
        self.video_encoder = VideoTokenizer(
            d_model=d_model,
            img_size=img_size,
            patch_size=patch_size,
            n_frames=n_frames,
            in_channels=video_channels
        )

        # Audio encoder (import from audio_tokenizer)
        from .audio_tokenizer import AudioTokenizer
        self.audio_encoder = AudioTokenizer(
            d_model=d_model,
            n_mels=n_mels,
            target_seq_len=audio_len // 10  # Downsample audio
        )

        # Fusion module
        if fusion_type == 'cross_attention':
            self.video_to_audio_attn = nn.MultiheadAttention(d_model, num_heads=8, batch_first=True)
            self.audio_to_video_attn = nn.MultiheadAttention(d_model, num_heads=8, batch_first=True)
        elif fusion_type == 'concat':
            # Concatenate and project back to d_model
            self.fusion_proj = nn.Linear(d_model * 2, d_model)

    def forward(
        self,
        video: torch.Tensor,
        audio: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            video: (batch, channels, frames, height, width)
            audio: (batch, n_mels, audio_len) or (batch, audio_len, n_mels)

        Returns:
            tokens: (batch, n_tokens, d_model)
        """
        # Encode video
        video_tokens = self.video_encoder(video)  # (B, T_v, d_model)

        # Encode audio
        audio_tokens = self.audio_encoder(audio)  # (B, T_a, d_model)

        # Fuse modalities
        if self.fusion_type == 'concat':
            # Align sequence lengths
            min_len = min(video_tokens.shape[1], audio_tokens.shape[1])
            video_tokens = video_tokens[:, :min_len, :]
            audio_tokens = audio_tokens[:, :min_len, :]

            # Concatenate along feature dimension
            fused = torch.cat([video_tokens, audio_tokens], dim=-1)
            fused = self.fusion_proj(fused)

        elif self.fusion_type == 'add':
            # Simple addition (requires aligned sequences)
            min_len = min(video_tokens.shape[1], audio_tokens.shape[1])
            fused = video_tokens[:, :min_len, :] + audio_tokens[:, :min_len, :]

        elif self.fusion_type == 'cross_attention':
            # Cross-modal attention
            video_attended, _ = self.audio_to_video_attn(
                query=video_tokens,
                key=audio_tokens,
                value=audio_tokens
            )
            audio_attended, _ = self.video_to_audio_attn(
                query=audio_tokens,
                key=video_tokens,
                value=video_tokens
            )

            # Concatenate
            fused = torch.cat([video_attended, audio_attended], dim=1)

        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")

        return fused


class BehaviorVideoTokenizer(nn.Module):
    """
    Specialized tokenizer for behavioral tracking videos.

    Optimized for:
    - Pose estimation outputs (keypoints)
    - Body part trajectories
    - Lower resolution behavioral videos
    """

    def __init__(
        self,
        d_model: int = 512,
        n_keypoints: int = 17,  # e.g., COCO keypoints
        n_frames: int = 100,
        use_cnn: bool = True,
        use_pose: bool = True
    ):
        """
        Args:
            d_model: Output dimension
            n_keypoints: Number of body keypoints
            n_frames: Number of frames
            use_cnn: Use CNN on raw video
            use_pose: Use pose keypoints if available
        """
        super().__init__()

        self.d_model = d_model
        self.n_keypoints = n_keypoints
        self.n_frames = n_frames
        self.use_cnn = use_cnn
        self.use_pose = use_pose

        if use_cnn:
            # Lightweight CNN for behavioral video
            self.cnn = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1))
            )
            self.cnn_proj = nn.Linear(128, d_model // 2 if use_pose else d_model)

        if use_pose:
            # Pose encoder: keypoint (x, y, confidence) -> embedding
            self.pose_encoder = nn.Sequential(
                nn.Linear(n_keypoints * 3, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, d_model // 2 if use_cnn else d_model)
            )

        # Temporal encoder (1D conv over time)
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
        )

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, n_frames, d_model))

    def forward(
        self,
        video: Optional[torch.Tensor] = None,
        keypoints: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            video: (batch, frames, channels, height, width) raw video
            keypoints: (batch, frames, n_keypoints, 3) pose keypoints (x, y, confidence)

        Returns:
            tokens: (batch, n_frames, d_model)
        """
        batch_size = video.shape[0] if video is not None else keypoints.shape[0]
        n_frames = video.shape[1] if video is not None else keypoints.shape[1]

        features = []

        # Process video with CNN
        if self.use_cnn and video is not None:
            # Flatten batch and time
            B, T, C, H, W = video.shape
            video_flat = video.reshape(B * T, C, H, W)

            # Apply CNN
            cnn_features = self.cnn(video_flat)  # (B*T, 128, 1, 1)
            cnn_features = cnn_features.squeeze(-1).squeeze(-1)  # (B*T, 128)
            cnn_features = self.cnn_proj(cnn_features)  # (B*T, d_model//2)
            cnn_features = cnn_features.reshape(B, T, -1)  # (B, T, d_model//2)

            features.append(cnn_features)

        # Process pose keypoints
        if self.use_pose and keypoints is not None:
            # Flatten keypoints
            B, T, K, D = keypoints.shape
            keypoints_flat = keypoints.reshape(B, T, K * D)

            # Encode
            pose_features = self.pose_encoder(keypoints_flat)  # (B, T, d_model//2)

            features.append(pose_features)

        # Combine features
        if len(features) == 2:
            combined = torch.cat(features, dim=-1)
        elif len(features) == 1:
            combined = features[0]
        else:
            raise ValueError("Must provide either video or keypoints")

        # Add positional embedding
        combined = combined + self.pos_embed[:, :n_frames, :]

        # Temporal processing
        combined = combined.transpose(1, 2)  # (B, d_model, T)
        combined = self.temporal_conv(combined)
        combined = combined.transpose(1, 2)  # (B, T, d_model)

        return combined
