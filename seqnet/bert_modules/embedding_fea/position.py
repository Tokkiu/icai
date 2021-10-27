import torch.nn as nn
import torch
import math


class PositionalEmbedding(nn.Module):

    def __init__(self, max_len, d_model):
        super().__init__()

        # Compute the positional encodings once in log space.
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x):
        b = self.pe.weight.unsqueeze(0).repeat(x.size(0) * x.size(2), 1, 1)
        c = b.reshape(x.size(0), -1, b.size(-2), b.size(-1))
        d = c.permute(0, 2, 1, 3)
        return d
