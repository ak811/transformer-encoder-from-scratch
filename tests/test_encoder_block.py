import torch

from encoder_block import TransformerEncoderBlock


def test_encoder_block_shapes():
    d_model = 64
    num_heads = 8
    d_ff = 256

    encoder_block = TransformerEncoderBlock(d_model, num_heads, d_ff)

    batch_size = 3
    seq_length = 7

    x = torch.randn(batch_size, seq_length, d_model)
    out = encoder_block(x)

    assert out.shape == (batch_size, seq_length, d_model)
