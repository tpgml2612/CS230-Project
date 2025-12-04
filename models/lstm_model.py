"""Many-to-one LSTM that ingests variable-length two-channel sequences."""
from typing import Optional

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class LSTM(nn.Module):
    def __init__(
        self,
        input_size: int = 2,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 1,
    ) -> None:
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        head_layers = [
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size)
        ]
        self.head = nn.Sequential(*head_layers)

    def forward(self, sequences: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        if lengths is None:
            raise ValueError("LSTM.forward requires sequence lengths when using packed sequences")
        packed = pack_padded_sequence(
            sequences, lengths.cpu(), batch_first=True, enforce_sorted=True
        )
        _, (hidden, _) = self.lstm(packed)
        final_hidden = hidden[-1]
        return self.head(final_hidden)
