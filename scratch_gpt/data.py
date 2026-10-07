import numpy as np


def encode_file(tok, txt_path, bin_path):
    """Tokenise a text file and store ids as uint16 (vocab < 65536)."""
    with open(txt_path, encoding="utf-8") as f:
        ids = tok.encode(f.read())
    np.array(ids, dtype=np.uint16).tofile(bin_path)
    return len(ids)


def load_bin(path):
    import torch
    return torch.from_numpy(np.fromfile(path, dtype=np.uint16).astype(np.int64))


def get_batch(data, batch_size, block_size, device):
    """Sample random contiguous windows; targets are the inputs shifted by one."""
    import torch
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)
