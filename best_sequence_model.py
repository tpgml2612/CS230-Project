"""Hybrid CNN + BiLSTM many-to-one regressor tailored for valid_examples_listseq.csv."""
import argparse
import ast
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils.data import DataLoader, Dataset


def _parse_list(value) -> List[float]:
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        return [float(v) for v in ast.literal_eval(value)]
    if isinstance(value, (list, tuple, np.ndarray)):
        return [float(v) for v in value]
    return [float(value)]


def load_sequences(examples_path: str, labels_path: str) -> Tuple[List[torch.Tensor], torch.Tensor]:
    examples_df = pd.read_csv(examples_path)
    labels_df = pd.read_csv(labels_path)
    if len(examples_df) != len(labels_df):
        raise ValueError("Examples and labels must have the same number of rows")

    sequences = []
    labels = []
    for idx in range(len(examples_df)):
        seq_a = torch.tensor(_parse_list(examples_df.iloc[idx, 0]), dtype=torch.float32)
        seq_b = torch.tensor(_parse_list(examples_df.iloc[idx, 1]), dtype=torch.float32)
        seq_len = min(seq_a.numel(), seq_b.numel())
        if seq_len == 0:
            continue
        stacked = torch.stack((seq_a[:seq_len], seq_b[:seq_len]), dim=-1)
        sequences.append(stacked)

        label_tensor = torch.tensor(labels_df.iloc[idx].values, dtype=torch.float32)
        if label_tensor.ndim == 0:
            label_tensor = label_tensor.unsqueeze(0)
        labels.append(label_tensor)

    if not sequences:
        raise ValueError("No valid sequences parsed from CSV files")

    labels_tensor = torch.stack(labels, dim=0)
    return sequences, labels_tensor


def compute_channel_stats(sequences: List[torch.Tensor], index_mask: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    values = [
        sequences[i]
        for i in index_mask
        if sequences[i].size(0) > 0
    ]
    if not values:
        return torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0])
    concat = torch.cat(values, dim=0)
    mean = concat.mean(dim=0)
    std = concat.std(dim=0, unbiased=False).clamp_min(1e-6)
    return mean, std


class TwoChannelDataset(Dataset):
    def __init__(
        self,
        sequences: List[torch.Tensor],
        labels: torch.Tensor,
        indices: np.ndarray,
        channel_mean: torch.Tensor,
        channel_std: torch.Tensor,
        downsample: int = 1,
    ):
        self.sequences = []
        self.labels = labels[indices]
        self.downsample = max(1, downsample)
        for idx in indices:
            seq = sequences[idx]
            if self.downsample > 1:
                seq = seq[::self.downsample]
            seq = (seq - channel_mean) / channel_std
            self.sequences.append(seq)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.labels[idx]


def lstm_collate_fn(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([seq.size(0) for seq in sequences], dtype=torch.long)
    order = torch.argsort(lengths, descending=True)
    sequences = [sequences[i] for i in order]
    labels = torch.stack([labels[i] for i in order]).float()
    lengths = lengths[order]
    padded = pad_sequence(sequences, batch_first=True)
    return padded, lengths, labels


class CNNBiLSTM(nn.Module):
    def __init__(
        self,
        conv_hidden: int = 64,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.2,
        output_dim: int = 2,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, conv_hidden, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        lstm_dropout = dropout if lstm_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=conv_hidden,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden * 2, lstm_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, output_dim),
        )

    def forward(self, sequences: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x = sequences.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)
        _, (hidden, _) = self.lstm(packed)
        final_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.head(final_hidden)


@dataclass
class TrainConfig:
    epochs: int = 40
    batch_size: int = 16
    lr: float = 3e-4
    val_split: float = 0.2
    downsample: int = 4
    seed: int = 42


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    n = 0
    for sequences, lengths, labels in loader:
        sequences = sequences.to(device)
        lengths = lengths.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        preds = model(sequences, lengths)
        loss = loss_fn(preds, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * sequences.size(0)
        n += sequences.size(0)
    return total_loss / max(1, n)


def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    n = 0
    preds_all = []
    trues_all = []
    with torch.no_grad():
        for sequences, lengths, labels in loader:
            sequences = sequences.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)
            preds = model(sequences, lengths)
            loss = loss_fn(preds, labels)
            total_loss += loss.item() * sequences.size(0)
            n += sequences.size(0)
            preds_all.append(preds.cpu())
            trues_all.append(labels.cpu())
    if preds_all:
        preds_all = torch.cat(preds_all)
        trues_all = torch.cat(trues_all)
    return total_loss / max(1, n), preds_all, trues_all


def run_training(args):
    torch.manual_seed(args.seed)
    sequences, labels = load_sequences(args.examples, args.labels)
    indices = np.arange(len(sequences))
    train_idx, val_idx = train_test_split(
        indices, test_size=args.val_split, random_state=args.seed, shuffle=True
    )

    channel_mean, channel_std = compute_channel_stats(sequences, train_idx)

    label_scaler = None
    if args.scale_labels:
        scaler = StandardScaler()
        scaler.fit(labels[train_idx])
        labels_scaled = torch.from_numpy(scaler.transform(labels.numpy())).float()
        labels = labels_scaled
        label_scaler = scaler

    train_dataset = TwoChannelDataset(
        sequences, labels, train_idx, channel_mean, channel_std, downsample=args.downsample
    )
    val_dataset = TwoChannelDataset(
        sequences, labels, val_idx, channel_mean, channel_std, downsample=args.downsample
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lstm_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lstm_collate_fn,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    model = CNNBiLSTM(
        conv_hidden=args.conv_hidden,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        dropout=args.dropout,
        output_dim=labels.size(1),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    best_val = float('inf')
    best_state = None
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, _, _ = evaluate(model, val_loader, loss_fn, device)
        if val_loss < best_val:
            best_val = val_loss
            best_state = model.state_dict()
        if epoch % 5 == 0 or epoch == 1 or epoch == args.epochs:
            print(f"Epoch {epoch}/{args.epochs} | Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(model.state_dict(), args.save_path)
        print(f"Saved best model to {args.save_path} (val loss={best_val:.4f})")

    val_loss, preds, trues = evaluate(model, val_loader, loss_fn, device)
    if label_scaler is not None and preds is not None:
        trues = torch.from_numpy(label_scaler.inverse_transform(trues.numpy()))
        preds = torch.from_numpy(label_scaler.inverse_transform(preds.numpy()))
    if preds is not None:
        mse = nn.functional.mse_loss(preds, trues).item()
        mae = nn.functional.l1_loss(preds, trues).item()
        print(f"Validation MSE (denormalized): {mse:.4f}, MAE: {mae:.4f}")


def build_argparser():
    parser = argparse.ArgumentParser(description="Train CNN+BiLSTM model on sequence CSV data.")
    parser.add_argument("--examples", default="valid_examples_listseq.csv")
    parser.add_argument("--labels", default="valid_labels_listseq.csv")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--downsample", type=int, default=4)
    parser.add_argument("--scale_labels", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--conv_hidden", type=int, default=64)
    parser.add_argument("--lstm_hidden", type=int, default=128)
    parser.add_argument("--lstm_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--save_path", default="best_sequence_model.pth")
    return parser


if __name__ == "__main__":
    args = build_argparser().parse_args()
    run_training(args)
