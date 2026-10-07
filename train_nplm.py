"""Train the NPLM baseline with the same optimizer, LR schedule, and token data as train.py,
so the only difference vs. the transformer run is architecture (fixed context + feedforward
vs. full context + attention), not optimization setup."""

import argparse
import math
import os
import time

import torch

from scratch_gpt.data import load_bin
from nplm import NPLM, NPLMConfig
from scratch_gpt.tokenizer import BPETokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--max_steps", type=int, default=3000)
ap.add_argument("--batch_size", type=int, default=32)
ap.add_argument("--context", type=int, default=8)
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--min_lr", type=float, default=2e-4)
ap.add_argument("--warmup", type=int, default=150)
ap.add_argument("--weight_decay", type=float, default=0.1)
ap.add_argument("--grad_clip", type=float, default=1.0)
ap.add_argument("--dropout", type=float, default=0.1)
ap.add_argument("--d_emb", type=int, default=160)
ap.add_argument("--d_hidden", type=int, default=861)
ap.add_argument("--eval_interval", type=int, default=250)
ap.add_argument("--eval_iters", type=int, default=40)
ap.add_argument("--seed", type=int, default=1337)
ap.add_argument("--out", default="checkpoints/nplm.pt")
args = ap.parse_args()

torch.manual_seed(args.seed)
device = "cuda" if torch.cuda.is_available() else "cpu"

tok = BPETokenizer.load("data/tokenizer.json")
train_data, val_data = load_bin("data/train.bin"), load_bin("data/val.bin")
cfg = NPLMConfig(
    vocab_size=tok.vocab_size,
    context=args.context,
    d_emb=args.d_emb,
    d_hidden=args.d_hidden,
    dropout=args.dropout,
)
model = NPLM(cfg).to(device)
print(
    f"device={device}  params={model.num_params() / 1e6:.2f}M  context={args.context}  "
    f"train_tokens={len(train_data):,}  val_tokens={len(val_data):,}"
)

decay = [p for p in model.parameters() if p.dim() >= 2]
no_decay = [p for p in model.parameters() if p.dim() < 2]
opt = torch.optim.AdamW(
    [
        {"params": decay, "weight_decay": args.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ],
    lr=args.lr,
    betas=(0.9, 0.95),
)


def lr_at(step):
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    t = (step - args.warmup) / max(1, args.max_steps - args.warmup)
    return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * t))


def get_batch(data, batch_size, context, device):
    """Sample random positions; input is the `context` tokens before it, target is the token itself."""
    ix = torch.randint(context, len(data) - 1, (batch_size,))
    x = torch.stack([data[i - context : i] for i in ix])
    y = torch.stack([data[i] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def evaluate():
    model.eval()
    out = {}
    for name, data in (("train", train_data), ("val", val_data)):
        losses = torch.zeros(args.eval_iters)
        for k in range(args.eval_iters):
            x, y = get_batch(data, args.batch_size, args.context, device)
            losses[k] = model(x, y)[1].item()
        out[name] = losses.mean().item()
    model.train()
    return out


os.makedirs(os.path.dirname(args.out), exist_ok=True)
best_val, t0 = float("inf"), time.time()
model.train()
for step in range(args.max_steps + 1):
    if step % args.eval_interval == 0 or step == args.max_steps:
        m = evaluate()
        print(
            f"step {step:5d} | train {m['train']:.3f} | val {m['val']:.3f} "
            f"| val ppl {math.exp(m['val']):.1f} | {time.time() - t0:.0f}s"
        )
        if m["val"] < best_val:
            best_val = m["val"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": cfg.to_dict(),
                    "val_loss": best_val,
                },
                args.out,
            )
    if step == args.max_steps:
        break
    for g in opt.param_groups:
        g["lr"] = lr_at(step)
    x, y = get_batch(train_data, args.batch_size, args.context, device)
    _, loss = model(x, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    opt.step()

print(f"best val loss {best_val:.3f} (ppl {math.exp(best_val):.1f}) -> {args.out}")
