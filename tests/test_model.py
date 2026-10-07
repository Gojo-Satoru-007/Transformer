import os, sys, math

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F
from scratch_gpt.model import GPT, GPTConfig
from scratch_gpt.rope import RotaryEmbedding
from scratch_gpt.attention import CausalSelfAttention


def test_rope_is_relative():
    """<R(m)q, R(n)k> must only depend on m - n."""
    rope = RotaryEmbedding(16, 64)
    q, k = torch.randn(1, 1, 1, 16), torch.randn(1, 1, 1, 16)

    def score(m, n):
        qm = rope(q.expand(1, 1, m + 1, 16).contiguous())[..., m, :]
        kn = rope(k.expand(1, 1, n + 1, 16).contiguous())[..., n, :]
        return (qm * kn).sum()

    assert torch.allclose(score(5, 2), score(25, 22), atol=1e-4)
    assert torch.allclose(score(10, 10), score(0, 0), atol=1e-4)


def test_rope_preserves_norm():
    rope = RotaryEmbedding(8, 32)
    x = torch.randn(2, 3, 32, 8)
    assert torch.allclose(rope(x).norm(dim=-1), x.norm(dim=-1), atol=1e-5)


def test_attention_matches_manual():
    torch.manual_seed(0)
    attn = CausalSelfAttention(32, 4, dropout=0.0).eval()
    rope = RotaryEmbedding(8, 16)
    x = torch.randn(2, 10, 32)
    B, T, C = x.shape
    q, k, v = attn.qkv(x).split(C, -1)
    q, k, v = (t.view(B, T, 4, 8).transpose(1, 2) for t in (q, k, v))
    q, k = rope(q), rope(k)
    s = q @ k.transpose(-1, -2) / math.sqrt(8)
    s = s.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool), 1), float("-inf"))
    ref = attn.proj((F.softmax(s, -1) @ v).transpose(1, 2).reshape(B, T, C))
    assert torch.allclose(attn(x, rope), ref, atol=1e-5)


def test_param_budget():
    n = GPT(GPTConfig()).num_params()
    assert 2e6 <= n <= 3e6, n


def test_causality():
    model = GPT(GPTConfig(n_layers=2, dropout=0.0)).eval()
    a = torch.randint(0, 10000, (1, 20))
    b = a.clone()
    b[0, 15:] = (b[0, 15:] + 1) % 10000
    la, lb = model(a)[0], model(b)[0]
    assert torch.allclose(la[:, :15], lb[:, :15], atol=1e-5)


def test_can_overfit_one_batch():
    torch.manual_seed(0)
    model = GPT(GPTConfig(n_layers=2, d_model=64, d_ff=128, dropout=0.0))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    x = torch.randint(0, 10000, (4, 32))
    y = torch.roll(x, -1, 1)
    first = model(x, y)[1].item()
    for _ in range(60):
        loss = model(x, y)[1]
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < first * 0.5
