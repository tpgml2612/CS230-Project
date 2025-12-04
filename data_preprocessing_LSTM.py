"""Preprocessing utilities for standard many-to-one LSTM training."""
import ast
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


def _parse_list(value: Sequence) -> List[float]:
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


class SequenceDataset(Dataset):
    def __init__(self, sequences: List[torch.Tensor], labels: torch.Tensor):
        if len(sequences) != labels.shape[0]:
            raise ValueError("Number of sequences and labels must match")
        self.sequences = sequences
        self.labels = labels.float()

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.labels[idx]


def lstm_collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([seq.size(0) for seq in sequences], dtype=torch.long)
    sorted_indices = torch.argsort(lengths, descending=True)
    sequences_sorted = [sequences[i] for i in sorted_indices]
    labels_sorted = torch.stack([labels[i] for i in sorted_indices]).float()
    lengths_sorted = lengths[sorted_indices]

    padded_sequences = pad_sequence(sequences_sorted, batch_first=True)
    return padded_sequences, lengths_sorted, labels_sorted


def preprocess_data_LSTM(
    examples_path: str,
    labels_path: str,
    batch_size: int,
    val_split: float = 0.2,
    seed: int = 42,
    shuffle: bool = True,
    num_workers: int = 0,
    scale_labels: bool = False,
    scale_inputs: bool = True,
    downsample_factor: int = 1,
) -> Tuple[DataLoader, DataLoader, int, Optional[StandardScaler]]:
    examples_df = pd.read_csv(examples_path)
    labels_df = pd.read_csv(labels_path)
    if len(examples_df) != len(labels_df):
        raise ValueError("Examples and labels CSVs must have the same number of rows")

    sequences: List[torch.Tensor] = []
    label_tensors: List[torch.Tensor] = []
    for idx in range(len(examples_df)):
        seq_a = torch.tensor(_parse_list(examples_df.iloc[idx, 0]), dtype=torch.float)
        seq_b = torch.tensor(_parse_list(examples_df.iloc[idx, 1]), dtype=torch.float)
        seq_len = min(seq_a.numel(), seq_b.numel())
        if seq_len == 0:
            continue
        seq_a = seq_a[:seq_len]
        seq_b = seq_b[:seq_len]
        if downsample_factor > 1:
            seq_a = seq_a[::downsample_factor]
            seq_b = seq_b[::downsample_factor]
            seq_len = min(seq_a.numel(), seq_b.numel())
            if seq_len == 0:
                continue
        stacked = torch.stack((seq_a[:seq_len], seq_b[:seq_len]), dim=-1)
        sequences.append(stacked)
        label_values = torch.tensor(labels_df.iloc[idx].values, dtype=torch.float)
        if label_values.ndim == 0:
            label_values = label_values.unsqueeze(0)
        label_tensors.append(label_values)

    if not sequences:
        raise ValueError("No valid sequences found after parsing CSV files.")

    labels_tensor = torch.stack(label_tensors, dim=0)
    indices = np.arange(len(sequences))
    train_idx, val_idx = train_test_split(
        indices, test_size=val_split, random_state=seed, shuffle=True
    )

    def _channel_stats(index_list: np.ndarray, channel: int) -> Tuple[torch.Tensor, torch.Tensor]:
        values = [
            sequences[i][:, channel]
            for i in index_list
            if sequences[i].size(0) > 0
        ]
        if not values:
            return torch.tensor(0.0), torch.tensor(1.0)
        concat = torch.cat(values)
        mean = concat.mean()
        std = concat.std(unbiased=False).clamp_min(1e-6)
        return mean, std

    if scale_inputs:
        mean_a, std_a = _channel_stats(train_idx, 0)
        mean_b, std_b = _channel_stats(train_idx, 1)
        for i in range(len(sequences)):
            seq = sequences[i]
            norm_a = (seq[:, 0] - mean_a) / std_a
            norm_b = (seq[:, 1] - mean_b) / std_b
            sequences[i] = torch.stack((norm_a, norm_b), dim=-1)

    if scale_labels:
        scaler = StandardScaler()
        labels_np = labels_tensor.numpy()
        scaler.fit(labels_np[train_idx])
        labels_scaled = torch.from_numpy(scaler.transform(labels_np)).float()
    else:
        scaler = None
        labels_scaled = labels_tensor.float()

    train_sequences = [sequences[i] for i in train_idx]
    val_sequences = [sequences[i] for i in val_idx]
    train_labels = labels_scaled[train_idx]
    val_labels = labels_scaled[val_idx]

    train_dataset = SequenceDataset(train_sequences, train_labels)
    val_dataset = SequenceDataset(val_sequences, val_labels)

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

    return train_loader, val_loader, 2, scaler
