import math
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding as in "Attention is All You Need".

    This implementation expects inputs of shape (batch_size, seq_length, d_model)
    and stores positional encodings of shape (1, max_len, d_model).
    """

    def __init__(self, d_model, max_len=5000):
        super().__init__()

        # position: [max_len, 1]
        position = torch.arange(0, max_len).unsqueeze(1).float()

        # div_term: [d_model/2]
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )

        # pe: [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # [1, max_len, d_model] so we can broadcast over batch
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_length, d_model)

        Returns:
            Tensor of same shape with positional encodings added.
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x


def save_positional_encoding_plot(d_model=20, max_len=100, out_path="assets/positional_encoding.png"):
    """
    Generate and save a visualization of positional encodings.
    """
    os.makedirs("assets", exist_ok=True)
    pos_encoding = PositionalEncoding(d_model, max_len)
    # pos_encoding.pe shape: [1, max_len, d_model]
    matrix = pos_encoding.pe[0, :, :].transpose(0, 1).numpy()  # [d_model, max_len]

    plt.figure(figsize=(10, 5))
    plt.pcolormesh(matrix, cmap="viridis")
    plt.xlabel("Position in sequence")
    plt.ylabel("Embedding dimension")
    plt.colorbar()
    plt.title("Positional encoding")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved positional encoding plot to {out_path}")


if __name__ == "__main__":
    save_positional_encoding_plot()
