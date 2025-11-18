import torch
import torch.nn as nn

from multi_head_attention import MultiHeadAttention


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout_rate=0.1):
        """
        Single Transformer encoder block.

        Args:
            d_model: Embedding dimension.
            num_heads: Number of attention heads.
            d_ff: Dimension of feed forward layer.
            dropout_rate: Dropout probability.
        """
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(d_model, num_heads)

        # Feed forward network: Linear -> ReLU -> Dropout -> Linear
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_ff, d_model),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_length, d_model)
            mask: Optional attention mask.

        Returns:
            Tensor of shape (batch_size, seq_length, d_model)
        """
        # Multi head self attention + residual + norm
        attention_output, _ = self.multi_head_attention(x, x, x, mask)
        x = x + self.dropout1(attention_output)
        x = self.norm1(x)

        # Feed forward + residual + norm
        ff_output = self.feed_forward(x)
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)

        return x


if __name__ == "__main__":
    # Simple test of the encoder block
    d_model = 512
    num_heads = 8
    d_ff = 2048

    encoder_block = TransformerEncoderBlock(d_model, num_heads, d_ff)

    seq_length = 60
    batch_size = 20
    dummy_input = torch.rand(batch_size, seq_length, d_model)

    output = encoder_block(dummy_input)
    print("Encoder output shape:", output.shape)  # Expected: (20, 60, 512)
