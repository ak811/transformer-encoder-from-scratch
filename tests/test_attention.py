import torch

from attention import scaled_dot_product_attention


def test_scaled_dot_product_attention_shapes():
    batch_size = 2
    seq_len_q = 3
    seq_len_k = 4
    d_k = 5

    q = torch.randn(batch_size, seq_len_q, d_k)
    k = torch.randn(batch_size, seq_len_k, d_k)
    v = torch.randn(batch_size, seq_len_k, d_k)

    output, attn = scaled_dot_product_attention(q, k, v)

    assert output.shape == (batch_size, seq_len_q, d_k)
    assert attn.shape == (batch_size, seq_len_q, seq_len_k)
