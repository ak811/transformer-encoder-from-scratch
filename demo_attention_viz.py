import os

import matplotlib.pyplot as plt
import torch

from multi_head_attention import MultiHeadAttention


def save_attention_example(
    d_model=16,
    num_heads=2,
    seq_length=6,
    batch_size=1,
    out_path="assets/example_attention.png",
):
    os.makedirs("assets", exist_ok=True)

    mha = MultiHeadAttention(d_model, num_heads)

    # Simple pattern input: random for demo
    dummy_input = torch.randn(batch_size, seq_length, d_model)

    _, attention_weights = mha(dummy_input, dummy_input, dummy_input)
    # attention_weights shape: (batch, num_heads, seq_len, seq_len)

    attn = attention_weights[0, 0].detach().numpy()  # first batch, first head

    plt.figure(figsize=(5, 4))
    plt.imshow(attn, cmap="viridis", aspect="auto")
    plt.colorbar()
    plt.xlabel("Key positions")
    plt.ylabel("Query positions")
    plt.title("Example attention weights (head 0)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved attention example plot to {out_path}")


if __name__ == "__main__":
    save_attention_example()
