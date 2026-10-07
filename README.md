# transformer-from-scratch

A decoder-only transformer language model built end-to-end without high-level framework
abstractions: **byte-pair-encoding tokenizer**, **RoPE**, **multi-head self-attention** and an
**AdamW training loop**, trained on the **Brown corpus** with a **10,000-token vocabulary** and
**~2.84M parameters**.

## What is hand-written vs. what is allowed from PyTorch

| Component | Implementation |
|---|---|
| BPE tokenizer (training, encode, decode, save/load) | pure Python, `scratch_gpt/tokenizer.py` |
| Rotary position embeddings | hand-written rotation, `scratch_gpt/rope.py` |
| Multi-head causal self-attention | `scratch_gpt/attention.py` (uses `F.scaled_dot_product_attention`) |
| RMSNorm, SwiGLU MLP, pre-norm blocks, weight-tied GPT | `scratch_gpt/model.py` |
| Training loop, LR schedule, grad clipping, eval, checkpoints | `train.py` |
| **Used from PyTorch** | `nn.Module`, `nn.Linear`, `nn.Embedding` (lookup table), `nn.Dropout`, `F.scaled_dot_product_attention`, `F.cross_entropy`, `torch.optim.AdamW` |
| **Not used** | `nn.Transformer*`, `nn.MultiheadAttention`, HF `transformers`, `tokenizers`, `tiktoken` |

## Architecture (defaults)

| Hyper-parameter | Value |
|---|---|
| vocab size | 10,000 (256 bytes + 9,743 merges + `<\|endoftext\|>`) |
| layers / heads / d_model | 4 / 4 / 160 (head_dim 40) |
| MLP | SwiGLU, hidden 432 |
| norm | pre-norm RMSNorm |
| positions | RoPE (base 10,000), no learned position table |
| context | 128 tokens |
| embeddings | input/output tied |
| **parameters** | **2,840,480** |

Parameter count: embedding 1.60M + 4 × (attention 102,400 + MLP 207,360 + norms 320) + final norm 160.

## Quick start

```bash
pip install -r requirements.txt

python prepare_data.py      # downloads Brown via NLTK, 95/5 document-level train/val split
python train_tokenizer.py   # trains BPE (vocab 10k) on train split, encodes train/val -> .bin
python train.py             # trains the model, saves best-val checkpoint
python generate.py --prompt "The president of the"

pytest -q                   # tokenizer, RoPE, attention, causality, overfit tests
```

Useful flags: `python train.py --max_steps 5000 --batch_size 64 --lr 1e-3 --dropout 0.2`

## Tokenizer results (measured)

Trained on 5.8M characters of Brown train split in ~7 s:

- 9,743 merges, vocabulary exactly 10,000
- 1,363,309 train tokens / 74,100 val tokens
- 4.27 characters per token; encode→decode is lossless (including non-ASCII)

## Training results

Brown is small (~1.4M tokens), so a 2.8M-parameter model will overfit if trained for too long;
the training script keeps the **best-validation** checkpoint and uses dropout + weight decay.
Record your run here:

| steps | train loss | val loss | val ppl |
|---|---|---|---|
| _fill in after `python train.py`_ | | | |

## Design notes

- **BPE** is byte-level (no unknown tokens). Training keeps a pair→word index and a lazy max-heap, so
  each merge only updates the words that contain it (about 7 s for 9.7k merges).
  Pre-tokenisation is a GPT-2-style regex so merges never cross word/punctuation boundaries.
- **RoPE** rotates (even, odd) channel pairs of Q and K. `tests/test_model.py` verifies that the
  Q·K score depends only on relative offset and that rotation preserves vector norm.
- **Attention** is checked against a manual softmax(QKᵀ/√d)V with a causal mask.
- **Brown text** is pre-tokenised ("word , word ."), so generated text keeps spaces before punctuation.
- Tokenizer is trained on the train split only, so validation tokens are truly held out.

## Layout

```
scratch_gpt/  tokenizer.py  rope.py  attention.py  model.py  data.py
prepare_data.py   train_tokenizer.py   train.py   generate.py
tests/
```
