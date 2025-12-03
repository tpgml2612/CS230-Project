"""Utilities for preparing variable-length two-channel sequences for LSTM training."""
import ast
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader


def _parse_list(value: Sequence) -> List[float]:
    """Safely parse a stringified Python list into a list of floats."""
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError) as exc:
            raise ValueError(f"Unable to parse list from value: {value}") from exc
    elif isinstance(value, (list, tuple, np.ndarray)):
        parsed = value
    else:
        parsed = [value]

    return [float(x) for x in parsed]


def create_sliding_windows(sequence: torch.Tensor, window_size: int, step_size: int) -> List[torch.Tensor]:
    """Generate 1D sliding windows for a single channel sequence."""
    if sequence.numel() == 0:
        return []
    if sequence.numel() < window_size:
        padded = torch.zeros(window_size, dtype=sequence.dtype)
        padded[: sequence.numel()] = sequence
        return [padded]

    windows = []
    for start in range(0, sequence.numel() - window_size + 1, step_size):
        windows.append(sequence[start : start + window_size])

    if not windows:
        windows.append(sequence[-window_size:])
    return windows


class SequenceWindowDataset(Dataset):
    """Dataset that stores per-example window tensors and labels."""

    def __init__(self, sequences: List[torch.Tensor], labels: torch.Tensor):
        if len(sequences) != labels.shape[0]:
            raise ValueError("Number of sequences and labels must match")
        self.sequences = sequences
        self.labels = labels.float()
        self.label_dim = self.labels.shape[1] if self.labels.ndim > 1 else 1

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.labels[idx]


def lstm_collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
    """Pad variable-length sequences for LSTM consumption and sort by descending length."""
    sequences, labels = zip(*batch)
    lengths = torch.tensor([seq.size(0) for seq in sequences], dtype=torch.long)
    sorted_indices = torch.argsort(lengths, descending=True)

    sorted_sequences = [sequences[i] for i in sorted_indices]
    sorted_labels = torch.stack([labels[i] for i in sorted_indices]).float()
    sorted_lengths = lengths[sorted_indices]

    padded_sequences = pad_sequence(sorted_sequences, batch_first=True)
    return padded_sequences, sorted_lengths, sorted_labels


def preprocess_data_LSTM(
    examples_path: str,
    labels_path: str,
    window_size: int,
    step_size: int,
    batch_size: int,
    val_split: float = 0.2,
    seed: int = 42,
    shuffle: bool = True,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, int]:
    """Prepare DataLoaders for LSTM training with sliding-window sequences."""
    examples_df = pd.read_csv(examples_path)
    labels_df = pd.read_csv(labels_path)
    if len(examples_df) != len(labels_df):
        raise ValueError("Examples and labels CSVs must have the same number of rows")

    sequences: List[torch.Tensor] = []
    label_tensors: List[torch.Tensor] = []
    skipped = 0
    for idx in range(len(examples_df)):
        seq_a = torch.tensor(_parse_list(examples_df.iloc[idx, 0]), dtype=torch.float)
        seq_b = torch.tensor(_parse_list(examples_df.iloc[idx, 1]), dtype=torch.float)

        windows_a = create_sliding_windows(seq_a, window_size, step_size)
        windows_b = create_sliding_windows(seq_b, window_size, step_size)
        num_windows = min(len(windows_a), len(windows_b))
        if num_windows == 0:
            skipped += 1
            continue

        stacked = torch.stack(
            [torch.stack((windows_a[w], windows_b[w]), dim=-1) for w in range(num_windows)], dim=0
        )
        sequences.append(stacked)
        label_values = torch.tensor(labels_df.iloc[idx].values, dtype=torch.float)
        if label_values.ndim == 0:
            label_values = label_values.unsqueeze(0)
        label_tensors.append(label_values)

    if not sequences:
        raise ValueError("No valid sequences produced. Check window/step sizes.")

    labels_tensor = torch.stack(label_tensors, dim=0)
    indices = np.arange(len(sequences))
    train_idx, val_idx = train_test_split(
        indices, test_size=val_split, random_state=seed, shuffle=True
    )

    train_sequences = [sequences[i] for i in train_idx]
    val_sequences = [sequences[i] for i in val_idx]
    train_labels = labels_tensor[train_idx]
    val_labels = labels_tensor[val_idx]

    train_dataset = SequenceWindowDataset(train_sequences, train_labels)
    val_dataset = SequenceWindowDataset(val_sequences, val_labels)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lstm_collate_fn,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lstm_collate_fn,
        num_workers=num_workers,
    )

    if skipped:
        print(f"Skipped {skipped} examples that could not form at least one window pair.")

    return train_loader, val_loader, 2
