"""Rotary Position Embedding (Su et al., 2021), implemented by hand.

Each (even, odd) pair of channels in a query/key head is rotated by an angle
theta_i * position, theta_i = base^(-2i/d). Because rotations compose, the dot
product q_m . k_n depends only on the relative offset (m - n).
"""
import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_len: int, base: float = 10000.0):
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        angles = torch.outer(torch.arange(max_len).float(), inv_freq)  # (T, d/2)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """x: (B, H, T, head_dim) -> rotated tensor of same shape."""
        T = x.size(-2)
        cos = self.cos[offset : offset + T]
        sin = self.sin[offset : offset + T]
        x1, x2 = x[..., 0::2], x[..., 1::2]
        out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
        return out.flatten(-2)
