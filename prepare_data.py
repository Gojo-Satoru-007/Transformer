"""Download the Brown corpus and write a document-level train/val split to data/."""
import os
import random

import nltk
from nltk.corpus import brown

from scratch_gpt.tokenizer import EOT_TOKEN

os.makedirs("data", exist_ok=True)
nltk.download("brown", quiet=True)

docs = []
for fid in sorted(brown.fileids()):
    # one paragraph per line, tokens joined with single spaces (Brown is pre-tokenised)
    paras = [" ".join(" ".join(s) for s in p) for p in brown.paras(fid)]
    docs.append("\n".join(paras))

random.Random(1337).shuffle(docs)
n_val = max(1, int(0.05 * len(docs)))          # hold out 5% of *documents*
splits = {"val": docs[:n_val], "train": docs[n_val:]}
for name, ds in splits.items():
    with open(f"data/{name}.txt", "w", encoding="utf-8") as f:
        f.write((EOT_TOKEN).join(ds) + EOT_TOKEN)
    print(f"{name}: {len(ds)} documents, {sum(len(d) for d in ds):,} characters")
