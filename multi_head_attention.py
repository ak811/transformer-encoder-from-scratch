import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        """
        Multi head attention module.

        Args:
            d_model: Total embedding size.
            num_heads: Number of attention heads (d_model must be divisible by num_heads).
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads

        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.final_projection = nn.Linear(d_model, d_model)

        self.scale = torch.sqrt(torch.FloatTensor([self.d_k]))

    def split_heads(self, x, batch_size):
        """
        Split the last dimension into (num_heads, depth) and permute.

        Input shape:  (batch_size, seq_length, d_model)
        Output shape: (batch_size, num_heads, seq_length, d_k)
        """
        x = x.view(batch_size, -1, self.num_heads, self.d_k)
        return x.permute(0, 2, 1, 3)

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query, key, value: Tensors of shape (batch_size, seq_length, d_model)
            mask: Optional tensor broadcastable to (batch_size, num_heads, seq_length, seq_length)

        Returns:
            output: (batch_size, seq_length, d_model)
            attention_weights: (batch_size, num_heads, seq_length, seq_length)
        """
        batch_size = query.size(0)

        # Linear projections
        query = self.query_projection(query)
        key = self.key_projection(key)
        value = self.value_projection(value)

        # Split to heads
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        # Scaled dot product attention
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attention_weights = F.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, value)

        # Concatenate heads
        attention_output = attention_output.permute(0, 2, 1, 3).contiguous()
        attention_output = attention_output.view(batch_size, -1, self.num_heads * self.d_k)

        # Final projection
        output = self.final_projection(attention_output)

        return output, attention_weights


if __name__ == "__main__":
    # Example instantiation and test
    d_model = 512
    num_heads = 8
    mha = MultiHeadAttention(d_model, num_heads)

    seq_length = 60
    batch_size = 20

    dummy_query = torch.randn(batch_size, seq_length, d_model)
    dummy_key = torch.randn(batch_size, seq_length, d_model)
    dummy_value = torch.randn(batch_size, seq_length, d_model)

    output, attention_weights = mha(dummy_query, dummy_key, dummy_value)
    print("Output shape:", output.shape)           # Expected: (20, 60, 512)
    print("Attention shape:", attention_weights.shape)  # Expected: (20, 8, 60, 60)
