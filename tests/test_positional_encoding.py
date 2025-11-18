import torch

from positional_encoding import PositionalEncoding


def test_positional_encoding_shape():
    d_model = 32
    max_len = 50
    pos_enc = PositionalEncoding(d_model, max_len=max_len)

    batch_size = 4
    seq_length = 10

    x = torch.zeros(batch_size, seq_length, d_model)
    out = pos_enc(x)

    assert out.shape == (batch_size, seq_length, d_model)
