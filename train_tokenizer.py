"""Train the byte-level BPE tokenizer (vocab 10,000) on the *training split only*."""
import argparse
import time

from scratch_gpt.data import encode_file
from scratch_gpt.tokenizer import BPETokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--vocab_size", type=int, default=10_000)
args = ap.parse_args()

text = open("data/train.txt", encoding="utf-8").read()
t0 = time.time()
tok = BPETokenizer().train(text, vocab_size=args.vocab_size)
print(f"trained {len(tok.merges)} merges in {time.time() - t0:.1f}s")
tok.save("data/tokenizer.json")

sample = "The Fulton County Grand Jury said Friday an investigation of Atlanta's recent primary election."
ids = tok.encode(sample)
print(len(ids), "tokens:", [tok.decode([i]) for i in ids])
assert tok.decode(ids) == sample

for split in ("train", "val"):
    n = encode_file(tok, f"data/{split}.txt", f"data/{split}.bin")
    print(f"{split}: {n:,} tokens")
n_chars = len(text)
print(f"compression: {n_chars / len(tok.encode(text)):.2f} chars/token (train)")
