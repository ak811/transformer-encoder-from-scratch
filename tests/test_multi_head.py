import torch

from multi_head_attention import MultiHeadAttention


def test_multi_head_attention_shapes():
    d_model = 64
    num_heads = 8
    mha = MultiHeadAttention(d_model, num_heads)

    batch_size = 4
    seq_length = 10

    q = torch.randn(batch_size, seq_length, d_model)
    k = torch.randn(batch_size, seq_length, d_model)
    v = torch.randn(batch_size, seq_length, d_model)

    output, attn = mha(q, k, v)

    assert output.shape == (batch_size, seq_length, d_model)
    assert attn.shape == (batch_size, num_heads, seq_length, seq_length)
