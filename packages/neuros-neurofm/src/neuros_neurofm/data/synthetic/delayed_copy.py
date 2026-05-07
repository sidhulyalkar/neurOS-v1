"""
Delayed Copy Dataset for ENGRAM-FMx.

Tests the model's ability to maintain information over long delays
and reproduce sequences accurately.

Task: Given a sequence followed by a delay period, reproduce the original sequence.
"""

import torch
from torch.utils.data import Dataset
from typing import Optional, Tuple


class DelayedCopyDataset(Dataset):
    """Delayed copy task dataset.

    Generates sequences of the form:
        [seq1, seq2, ..., seqN, DELAY..., COPY_SIGNAL, ???]

    The model must reproduce the original sequence after the delay.

    Parameters
    ----------
    num_samples : int
        Number of samples to generate. Default: 10000.
    seq_length : int
        Total sequence length. Default: 256.
    copy_length : int
        Length of sequence to copy. Default: 16.
    delay_length : int
        Length of delay period. Default: 64.
    vocab_size : int
        Size of token vocabulary. Default: 32.
    hidden_dim : int
        Embedding dimension. Default: 128.
    seed : int, optional
        Random seed for reproducibility.
    """

    def __init__(
        self,
        num_samples: int = 10000,
        seq_length: int = 256,
        copy_length: int = 16,
        delay_length: int = 64,
        vocab_size: int = 32,
        hidden_dim: int = 128,
        seed: Optional[int] = None,
    ):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.copy_length = copy_length
        self.delay_length = delay_length
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

        if seed is not None:
            torch.manual_seed(seed)

        # Token embeddings
        self.token_embeddings = torch.randn(vocab_size, hidden_dim)
        self.token_embeddings = self.token_embeddings / self.token_embeddings.norm(dim=-1, keepdim=True)

        # Special tokens
        self.delay_token = torch.randn(hidden_dim) * 0.1  # Low magnitude for delay
        self.copy_signal = torch.randn(hidden_dim)
        self.copy_signal = self.copy_signal / self.copy_signal.norm()

        # Pre-generate samples
        self._generate_samples()

    def _generate_samples(self):
        """Pre-generate all samples."""
        self.samples = []

        for _ in range(self.num_samples):
            # Generate random sequence to copy
            sequence = torch.randint(0, self.vocab_size, (self.copy_length,))
            self.samples.append(sequence)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a sample.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            - input_sequence: [seq_length, hidden_dim]
            - target_sequence: [seq_length, hidden_dim]
            - mask: [seq_length] - True where prediction should be made
        """
        sequence = self.samples[idx]

        input_seq = torch.zeros(self.seq_length, self.hidden_dim)
        target_seq = torch.zeros(self.seq_length, self.hidden_dim)
        mask = torch.zeros(self.seq_length, dtype=torch.bool)

        # Phase 1: Original sequence
        for t in range(self.copy_length):
            if t >= self.seq_length:
                break
            input_seq[t] = self.token_embeddings[sequence[t]]

        # Phase 2: Delay period
        delay_start = self.copy_length
        delay_end = min(delay_start + self.delay_length, self.seq_length)
        for t in range(delay_start, delay_end):
            input_seq[t] = self.delay_token

        # Phase 3: Copy signal
        copy_signal_pos = delay_end
        if copy_signal_pos < self.seq_length:
            input_seq[copy_signal_pos] = self.copy_signal

        # Phase 4: Target sequence (model should predict this)
        output_start = copy_signal_pos + 1
        for i, t in enumerate(range(output_start, min(output_start + self.copy_length, self.seq_length))):
            target_seq[t] = self.token_embeddings[sequence[i]]
            mask[t] = True

        return input_seq, target_seq, mask

    @staticmethod
    def collate_fn(batch):
        """Collate function for DataLoader."""
        inputs = torch.stack([b[0] for b in batch])
        targets = torch.stack([b[1] for b in batch])
        masks = torch.stack([b[2] for b in batch])
        return inputs, targets, masks
