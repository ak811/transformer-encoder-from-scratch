import torch
import torch.nn as nn

from encoder_block import TransformerEncoderBlock
from positional_encoding import PositionalEncoding


class SimpleTransformerEncoder(nn.Module):
    """
    Minimal Transformer encoder stack:
    - Token embedding
    - Positional encoding
    - One or more TransformerEncoderBlock layers
    """

    def __init__(self, vocab_size, d_model=128, num_heads=4, d_ff=512, num_layers=2, max_len=100):
        super().__init__()
        self.d_model = d_model

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=max_len)

        self.layers = nn.ModuleList(
            [
                TransformerEncoderBlock(d_model, num_heads, d_ff)
                for _ in range(num_layers)
            ]
        )

    def forward(self, input_ids, mask=None):
        """
        Args:
            input_ids: LongTensor of shape (batch_size, seq_length)
            mask: Optional mask for attention.

        Returns:
            Tensor of shape (batch_size, seq_length, d_model)
        """
        x = self.token_embedding(input_ids) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        x = self.positional_encoding(x)

        for layer in self.layers:
            x = layer(x, mask=mask)

        return x


if __name__ == "__main__":
    # Demo usage of the simple encoder
    vocab_size = 1000
    batch_size = 2
    seq_length = 10

    model = SimpleTransformerEncoder(
        vocab_size=vocab_size,
        d_model=128,
        num_heads=4,
        d_ff=512,
        num_layers=2,
        max_len=50,
    )

    # Fake token ids
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))

    output = model(input_ids)
    print("Input ids shape:", input_ids.shape)   # (2, 10)
    print("Encoder output shape:", output.shape)  # (2, 10, 128)
