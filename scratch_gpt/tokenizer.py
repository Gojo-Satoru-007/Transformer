"""Byte-level Byte-Pair-Encoding tokenizer, written from scratch (no `tokenizers`/`tiktoken`).

Vocabulary layout (vocab_size = 10_000 by default):
    ids 0..255                -> raw bytes
    ids 256..vocab_size-2     -> learned merges, in the order they were learned
    id  vocab_size-1          -> <|endoftext|>
"""

import heapq
from itertools import pairwise
import json
import re
from collections import Counter, defaultdict

EOT_TOKEN = "<|endoftext|>"

# GPT-2 style pre-tokenisation (stdlib `re` version): contractions, words, numbers,
# punctuation runs and whitespace. Merges never cross these boundaries.
PRETOKENIZE = re.compile(
    r"""'(?:s|t|re|ve|m|ll|d)| ?[^\W\d_]+| ?\d+| ?[^\s\w]+|_|\s+(?!\S)|\s+"""
)


class BPETokenizer:
    def __init__(self, merges=None, vocab_size=10_000):
        self.merges = [tuple(m) for m in (merges or [])]
        self.vocab_size = vocab_size
        self._build()

    # ------------------------------------------------------------------ build
    def _build(self):
        self.ranks = {pair: i for i, pair in enumerate(self.merges)}
        self.vocab = {i: bytes([i]) for i in range(256)}
        for i, (a, b) in enumerate(self.merges):
            self.vocab[256 + i] = self.vocab[a] + self.vocab[b]
        self.eot_id = self.vocab_size - 1
        self._cache = {}

    # --------------------------------------------------------------- training
    def train(self, text: str, vocab_size: int = 10_000, verbose: bool = True):
        """Learn `vocab_size - 257` merges. Uses a pair->words index and a lazy heap so each
        merge only touches the words that actually contain the pair."""
        self.vocab_size = vocab_size
        n_merges = vocab_size - 256 - 1  # minus the special token
        text = text.replace(EOT_TOKEN, "\n")
        word_freq = Counter(
            m.group(0).encode("utf-8") for m in PRETOKENIZE.finditer(text)
        )
        words = [list(w) for w in word_freq]
        freqs = list(word_freq.values())

        pair_counts = defaultdict(int)
        where = defaultdict(set)
        for i, w in enumerate(words):
            for p in pairwise(w):
                pair_counts[p] += freqs[i]
                where[p].add(i)
        heap = [(-c, p) for p, c in pair_counts.items()]
        heapq.heapify(heap)

        merges = []
        while len(merges) < n_merges and heap:
            neg, best = heapq.heappop(heap)
            if pair_counts.get(best, 0) != -neg:  # stale heap entry
                continue
            new_id = 256 + len(merges)
            merges.append(best)
            changed = set()
            for i in list(where[best]):
                w, f = words[i], freqs[i]
                for p in pairwise(w):  # remove old pair contributions
                    pair_counts[p] -= f
                    changed.add(p)
                out, j = [], 0  # apply the merge
                while j < len(w):
                    if j < len(w) - 1 and (w[j], w[j + 1]) == best:
                        out.append(new_id)
                        j += 2
                    else:
                        out.append(w[j])
                        j += 1
                words[i] = out
                for p in pairwise(out):  # add new pair contributions
                    pair_counts[p] += f
                    where[p].add(i)
                    changed.add(p)
            for p in changed:
                c = pair_counts.get(p, 0)
                if c > 0:
                    heapq.heappush(heap, (-c, p))
                else:
                    pair_counts.pop(p, None)
            pair_counts.pop(best, None)
            if verbose and len(merges) % 1000 == 0:
                print(f"  merges: {len(merges)}/{n_merges}")
        self.merges = merges
        self._build()
        return self

    # --------------------------------------------------------------- encoding
    def _encode_word(self, word: bytes):
        if word in self._cache:
            return self._cache[word]
        ids = list(word)
        while len(ids) > 1:
            best, best_rank = None, None
            for p in pairwise(ids):
                r = self.ranks.get(p)
                if r is not None and (best_rank is None or r < best_rank):
                    best, best_rank = p, r
            if best_rank is None or best is None:
                break
            new_id, out, j = 256 + best_rank, [], 0
            while j < len(ids):
                if j < len(ids) - 1 and (ids[j], ids[j + 1]) == best:
                    out.append(new_id)
                    j += 2
                else:
                    out.append(ids[j])
                    j += 1
            ids = out
        self._cache[word] = ids
        return ids

    def encode(self, text: str):
        ids = []
        for k, chunk in enumerate(text.split(EOT_TOKEN)):
            if k > 0:
                ids.append(self.eot_id)
            for m in PRETOKENIZE.finditer(chunk):
                ids.extend(self._encode_word(m.group(0).encode("utf-8")))
        return ids

    def decode(self, ids):
        out = bytearray()
        for i in ids:
            out += EOT_TOKEN.encode() if i == self.eot_id else self.vocab[i]
        return out.decode("utf-8", errors="replace")

    # ---------------------------------------------------------------- save/io
    def save(self, path):
        with open(path, "w") as f:
            json.dump({"vocab_size": self.vocab_size, "merges": self.merges}, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        return cls(d["merges"], d["vocab_size"])
