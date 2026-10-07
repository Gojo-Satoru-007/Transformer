"""Training loop: AdamW + linear warm-up / cosine decay + grad clipping, on tokenised Brown."""
import argparse
import math
import os
import time

import torch

from scratch_gpt.data import get_batch, load_bin
from scratch_gpt.model import GPT, GPTConfig
from scratch_gpt.tokenizer import BPETokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--max_steps", type=int, default=3000)
ap.add_argument("--batch_size", type=int, default=32)
ap.add_argument("--block_size", type=int, default=128)
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--min_lr", type=float, default=2e-4)
ap.add_argument("--warmup", type=int, default=150)
ap.add_argument("--weight_decay", type=float, default=0.1)
ap.add_argument("--grad_clip", type=float, default=1.0)
ap.add_argument("--dropout", type=float, default=0.1)
ap.add_argument("--n_layers", type=int, default=4)
ap.add_argument("--n_heads", type=int, default=4)
ap.add_argument("--d_model", type=int, default=160)
ap.add_argument("--d_ff", type=int, default=432)
ap.add_argument("--eval_interval", type=int, default=250)
ap.add_argument("--eval_iters", type=int, default=40)
ap.add_argument("--seed", type=int, default=1337)
ap.add_argument("--out", default="checkpoints/ckpt.pt")
args = ap.parse_args()

torch.manual_seed(args.seed)
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

tok = BPETokenizer.load("data/tokenizer.json")
train_data, val_data = load_bin("data/train.bin"), load_bin("data/val.bin")
cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=args.block_size, n_layers=args.n_layers,
                n_heads=args.n_heads, d_model=args.d_model, d_ff=args.d_ff, dropout=args.dropout)
model = GPT(cfg).to(device)
print(f"device={device}  params={model.num_params() / 1e6:.2f}M  "
      f"train_tokens={len(train_data):,}  val_tokens={len(val_data):,}")

# AdamW: decay matrices (linear / embedding weights), not norm gains
decay = [p for p in model.parameters() if p.dim() >= 2]
no_decay = [p for p in model.parameters() if p.dim() < 2]
opt = torch.optim.AdamW(
    [{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
    lr=args.lr, betas=(0.9, 0.95),
)


def lr_at(step):
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    t = (step - args.warmup) / max(1, args.max_steps - args.warmup)
    return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * t))


@torch.no_grad()
def evaluate():
    model.eval()
    out = {}
    for name, data in (("train", train_data), ("val", val_data)):
        losses = torch.zeros(args.eval_iters)
        for k in range(args.eval_iters):
            x, y = get_batch(data, args.batch_size, args.block_size, device)
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
        print(f"step {step:5d} | train {m['train']:.3f} | val {m['val']:.3f} "
              f"| val ppl {math.exp(m['val']):.1f} | {time.time() - t0:.0f}s")
        if m["val"] < best_val:
            best_val = m["val"]
            torch.save({"model": model.state_dict(), "config": cfg.to_dict(), "val_loss": best_val}, args.out)
    if step == args.max_steps:
        break
    for g in opt.param_groups:
        g["lr"] = lr_at(step)
    x, y = get_batch(train_data, args.batch_size, args.block_size, device)
    _, loss = model(x, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    opt.step()

print(f"best val loss {best_val:.3f} (ppl {math.exp(best_val):.1f}) -> {args.out}")
