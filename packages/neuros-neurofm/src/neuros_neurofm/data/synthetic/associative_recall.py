"""
Associative Recall Dataset for ENGRAM-FMx.

Tests the model's ability to store and retrieve key-value pairs
from its attractor memory system.

Task: Given a sequence of key-value pairs followed by a query key,
predict the corresponding value.
"""

import torch
from torch.utils.data import Dataset
from typing import Optional, Tuple
import math


class AssociativeRecallDataset(Dataset):
    """Associative recall task dataset.

    Generates sequences of the form:
        [key1, val1, key2, val2, ..., keyN, valN, QUERY, keyQ, ANSWER]

    The model must retrieve the value associated with keyQ.

    Parameters
    ----------
    num_samples : int
        Number of samples to generate. Default: 10000.
    seq_length : int
        Total sequence length. Default: 256.
    vocab_size : int
        Size of key/value vocabulary. Default: 64.
    hidden_dim : int
        Embedding dimension. Default: 128.
    num_pairs : int
        Number of key-value pairs per sample. Default: 8.
    seed : int, optional
        Random seed for reproducibility.
    """

    def __init__(
        self,
        num_samples: int = 10000,
        seq_length: int = 256,
        vocab_size: int = 64,
        hidden_dim: int = 128,
        num_pairs: int = 8,
        seed: Optional[int] = None,
    ):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_pairs = num_pairs

        if seed is not None:
            torch.manual_seed(seed)

        # Pre-generate embeddings for keys and values
        # Use orthogonal initialization for better separability
        self.key_embeddings = torch.randn(vocab_size, hidden_dim)
        self.key_embeddings = self.key_embeddings / self.key_embeddings.norm(dim=-1, keepdim=True)

        self.value_embeddings = torch.randn(vocab_size, hidden_dim)
        self.value_embeddings = self.value_embeddings / self.value_embeddings.norm(dim=-1, keepdim=True)

        # Special tokens
        self.query_token = torch.randn(hidden_dim)
        self.query_token = self.query_token / self.query_token.norm()

        self.pad_token = torch.zeros(hidden_dim)

        # Pre-generate all samples for efficiency
        self._generate_samples()

    def _generate_samples(self):
        """Pre-generate all samples."""
        self.samples = []

        for _ in range(self.num_samples):
            # Sample unique keys
            keys = torch.randperm(self.vocab_size)[:self.num_pairs]
            values = torch.randint(0, self.vocab_size, (self.num_pairs,))

            # Choose query key (one of the stored keys)
            query_idx = torch.randint(0, self.num_pairs, (1,)).item()
            query_key = keys[query_idx]
            target_value = values[query_idx]

            # Build sequence
            # Structure: [k1, v1, k2, v2, ..., kN, vN, QUERY, kQ, padding...]
            seq_tokens = []
            for i in range(self.num_pairs):
                seq_tokens.append(("key", keys[i].item()))
                seq_tokens.append(("value", values[i].item()))

            seq_tokens.append(("query", None))
            seq_tokens.append(("key", query_key.item()))

            self.samples.append({
                "tokens": seq_tokens,
                "target_value": target_value.item(),
                "query_position": len(seq_tokens) - 1,
            })

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
        sample = self.samples[idx]

        # Build input sequence
        input_seq = torch.zeros(self.seq_length, self.hidden_dim)
        target_seq = torch.zeros(self.seq_length, self.hidden_dim)
        mask = torch.zeros(self.seq_length, dtype=torch.bool)

        for t, (token_type, token_id) in enumerate(sample["tokens"]):
            if t >= self.seq_length:
                break

            if token_type == "key":
                input_seq[t] = self.key_embeddings[token_id]
            elif token_type == "value":
                input_seq[t] = self.value_embeddings[token_id]
            elif token_type == "query":
                input_seq[t] = self.query_token

        # Target: at the query position, predict the value
        query_pos = sample["query_position"]
        if query_pos < self.seq_length:
            target_seq[query_pos] = self.value_embeddings[sample["target_value"]]
            mask[query_pos] = True

        return input_seq, target_seq, mask

    @staticmethod
    def collate_fn(batch):
        """Collate function for DataLoader."""
        inputs = torch.stack([b[0] for b in batch])
        targets = torch.stack([b[1] for b in batch])
        masks = torch.stack([b[2] for b in batch])
        return inputs, targets, masks
