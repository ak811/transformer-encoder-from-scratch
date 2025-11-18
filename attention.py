import torch
import torch.nn.functional as F


def scaled_dot_product_attention(q, k, v, mask=None):
    """
    Compute scaled dot product attention.

    Args:
        q: Query tensor (..., seq_len_q, d_k)
        k: Key tensor   (..., seq_len_k, d_k)
        v: Value tensor (..., seq_len_v, d_v)
        mask: Optional mask tensor broadcastable to (..., seq_len_q, seq_len_k)

    Returns:
        output: Attention output tensor (..., seq_len_q, d_v)
        attention_weights: Attention weights tensor (..., seq_len_q, seq_len_k)
    """
    d_k = q.size(-1)
    scale = torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

    scores = torch.matmul(q, k.transpose(-2, -1)) / scale

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    attention_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, v)
    return output, attention_weights


if __name__ == "__main__":
    # Simple sanity check to reproduce the lab behavior
    torch.manual_seed(42)
    seq_len, d_k = 3, 2
    q = torch.randn(seq_len, d_k)
    k = torch.randn(seq_len, d_k)
    v = torch.randn(seq_len, d_k)

    values, attention = scaled_dot_product_attention(q, k, v)

    print("Q\n", q)
    print("K\n", k)
    print("V\n", v)
    print("Values\n", values)
    print("Attention\n", attention)
