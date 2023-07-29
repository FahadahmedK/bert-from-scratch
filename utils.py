import math
import torch
import torch.nn as nn


def masked_softmax(X, valid_lens):
    def _sequence_mask(X, valid_lens, value=0):

        original_shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, repeats=original_shape[1], dim=0)
        else:
            valid_lens = valid_lens.reshape(-1)

        X = X.reshape(-1, X.shape[-1])  # (batch_size * maxlen, d)
        maxlen = X.size(1)
        mask = torch.arange(maxlen, dtype=torch.float32, device=X.device)[None, :] < valid_lens[:, None]

        X[~mask] = value
        return X.reshape(original_shape)

    if valid_lens is None:
        return torch.nn.functional.softmax(X, dim=-1)

    else:
        # fill masked elements with a large negative, whose exp is 0
        X = _sequence_mask(X, valid_lens, value=-1e6)
        X = torch.nn.functional.softmax(X, dim=-1)
        X = _sequence_mask(X, valid_lens, value=0)
        return X


class DotProductAttention(nn.Module):
    """Scaled dot product attention."""

    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        attention_weights = self.dropout(masked_softmax(scores, valid_lens))
        return torch.bmm(attention_weights, values)


class MultiHeadAttention(nn.Module):
    def __init__(self, num_hiddens, num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(num_hiddens, num_hiddens, bias=bias)
        self.W_k = nn.Linear(num_hiddens, num_hiddens, bias=bias)
        self.W_v = nn.Linear(num_hiddens, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens * num_heads, num_hiddens, bias=bias)

    def transpose_qkv(self, X):
        X = X.reshape(X.shape[0], X.shape[1], self.num_heads, -1)
        X = X.permute(0, 2, 1, 3)
        return X.reshape(-1, X.shape[2], X.shape[3])

    def transpose_output(self, X):
        X = X.reshape(-1, self.num_heads, X.shape[1], X.shape[2])
        X = X.permute(0, 2, 3, 1)
        return X.reshape(X.shape[0], X.shape[1], -1)

    def forward(self, queries, keys, values, valid_lens):
        queries = self.transpose_qkv(self.W_q(queries))
        keys = self.transpose_qkv(self.W_k(keys))
        values = self.transpose_qkv(self.W_v(values))

        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)

        output = self.attention(queries, keys, values, valid_lens)
        output_concat = self.transpose_output(output)

        return self.W_o(output_concat)

