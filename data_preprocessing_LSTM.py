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


def fast_lstm_collate_fn(batch):
    sequences, labels = zip(*batch)

    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
    padded = pad_sequence(sequences, batch_first=True)

    labels = torch.stack(labels).float()
    return padded, lengths, labels


def preprocess_data_LSTM(
    examples_path,
    labels_path,
    batch_size,
    val_split=0.2,
    seed=42,
    shuffle=True,
    scale_labels=False,
    scale_inputs=True,
    downsample_factor=1,
):
    examples_df = pd.read_csv(examples_path)
    labels_df = pd.read_csv(labels_path)

    sequences = []
    labels = []

    for idx in range(len(examples_df)):
        seq_a = torch.tensor(_parse_list(examples_df.iloc[idx, 0]), dtype=torch.float)
        seq_b = torch.tensor(_parse_list(examples_df.iloc[idx, 1]), dtype=torch.float)

        seq_len = min(len(seq_a), len(seq_b))
        if seq_len == 0:
            continue

        if downsample_factor > 1:
            seq_a = seq_a[::downsample_factor]
            seq_b = seq_b[::downsample_factor]
            seq_len = min(len(seq_a), len(seq_b))
            if seq_len == 0:
                continue

        stacked = torch.stack([seq_a[:seq_len], seq_b[:seq_len]], dim=-1)
        sequences.append(stacked)

        label_tensor = torch.tensor(labels_df.iloc[idx].values, dtype=torch.float)
        labels.append(label_tensor)

    labels_tensor = torch.stack(labels, dim=0)

    # train/val split
    idx_all = np.arange(len(sequences))
    train_idx, val_idx = train_test_split(idx_all, test_size=val_split, random_state=seed)

    # ==== FAST SCALING (vectorized) ====
    if scale_inputs:
        all_train_a = torch.cat([sequences[i][:, 0] for i in train_idx])
        all_train_b = torch.cat([sequences[i][:, 1] for i in train_idx])

        mean_a, std_a = all_train_a.mean(), all_train_a.std().clamp_min(1e-6)
        mean_b, std_b = all_train_b.mean(), all_train_b.std().clamp_min(1e-6)

        for i in range(len(sequences)):
            seq = sequences[i]
            seq[:, 0] = (seq[:, 0] - mean_a) / std_a
            seq[:, 1] = (seq[:, 1] - mean_b) / std_b

    # ==== Label scaling (same as before) ====
    if scale_labels:
        scaler = StandardScaler()
        scaler.fit(labels_tensor[train_idx])
        labels_scaled = torch.tensor(scaler.transform(labels_tensor), dtype=torch.float)
    else:
        scaler = None
        labels_scaled = labels_tensor

    # Create DataLoaders
    train_dataset = SequenceDataset([sequences[i] for i in train_idx], labels_scaled[train_idx])
    val_dataset   = SequenceDataset([sequences[i] for i in val_idx], labels_scaled[val_idx])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=fast_lstm_collate_fn)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=fast_lstm_collate_fn)

    return train_loader, val_loader, 2, scaler

