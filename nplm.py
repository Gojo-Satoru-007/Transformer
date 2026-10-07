"""Neural Probabilistic Language Model (Bengio et al., 2003), used as a fixed-context
feedforward baseline against the transformer.

Architecture: concatenate the embeddings of the CONTEXT preceding tokens, pass through
one tanh hidden layer, project back to d_emb, and compute logits via the (tied) token
embedding matrix -- the same tying convention the transformer uses, so the comparison
isolates "fixed window + feedforward" vs. "full context + attention", not embedding-tying
differences. Hidden size is chosen so the total parameter count matches the transformer's
2,840,480 as closely as integer arithmetic allows (2,840,861 here).
"""
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class NPLMConfig:
    vocab_size: int = 10_000
    context: int = 8        # fixed window of preceding tokens (Bengio et al. used n-1=4..6)
    d_emb: int = 160         # matches transformer d_model, for a fair embedding-table size
    d_hidden: int = 861      # calibrated so total params ~= transformer's 2,840,480
    dropout: float = 0.1

    def to_dict(self):
        return asdict(self)


class NPLM(nn.Module):
    def __init__(self, cfg: NPLMConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_emb)
        self.in_drop = nn.Dropout(cfg.dropout)
        self.hidden = nn.Linear(cfg.context * cfg.d_emb, cfg.d_hidden)
        self.out_proj = nn.Linear(cfg.d_hidden, cfg.d_emb)
        self.out_drop = nn.Dropout(cfg.dropout)
        self.lm_head = nn.Linear(cfg.d_emb, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # tied, same convention as the transformer
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())  # tied weights counted once

    def forward(self, ctx_idx, targets=None):
        """ctx_idx: (B, context) preceding-token ids. targets: (B,) next-token ids."""
        B = ctx_idx.size(0)
        x = self.in_drop(self.tok_emb(ctx_idx)).view(B, -1)      # (B, context*d_emb)
        h = torch.tanh(self.hidden(x))                            # (B, d_hidden)
        y = self.out_drop(self.out_proj(h))                       # (B, d_emb)
        logits = self.lm_head(y)                                  # (B, vocab)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=40):
        self.eval()
        C = self.cfg.context
        for _ in range(max_new_tokens):
            ctx = idx[:, -C:]
            if ctx.size(1) < C:  # left-pad with id 0 if the prompt is shorter than the context
                pad = torch.zeros(ctx.size(0), C - ctx.size(1), dtype=ctx.dtype, device=ctx.device)
                ctx = torch.cat([pad, ctx], dim=1)
            logits, _ = self(ctx)
            logits = logits / max(temperature, 1e-5)
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            nxt = torch.multinomial(F.softmax(logits, -1), 1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx
