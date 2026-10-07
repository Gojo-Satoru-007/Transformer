import argparse

import torch

from scratch_gpt.model import GPT, GPTConfig
from scratch_gpt.tokenizer import BPETokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--prompt", default="The president of the")
ap.add_argument("--max_new_tokens", type=int, default=80)
ap.add_argument("--temperature", type=float, default=0.8)
ap.add_argument("--top_k", type=int, default=40)
ap.add_argument("--ckpt", default="checkpoints/ckpt.pt")
args = ap.parse_args()

tok = BPETokenizer.load("data/tokenizer.json")
ckpt = torch.load(args.ckpt, map_location="cpu")
model = GPT(GPTConfig(**ckpt["config"]))
model.load_state_dict(ckpt["model"])
idx = torch.tensor([tok.encode(args.prompt)])
out = model.generate(idx, args.max_new_tokens, args.temperature, args.top_k)
print(tok.decode(out[0].tolist()))
