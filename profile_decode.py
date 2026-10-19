import time

import torch
import torch.nn.functional as F

from blocks import BlockAllocator, KVPool
from model import SmolLM2, paged_attention, rms_norm, rope_tables

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
CASES = [(1, 64), (1, 512), (8, 64), (28, 64), (28, 256), (28, 512)]
REPEATS = 20


def gpu_wait():
    """Block until the GPU has finished its queued work."""
    if DEVICE == "mps":
        torch.mps.synchronize()


def setup(model, users, length):
    """A pool with this many users, each holding length tokens of random K and V."""
    c = model.config
    blocks_each = length // 16 + 1
    allocator = BlockAllocator(users * blocks_each + 1)
    pool = KVPool(c.layers, c.kv_heads, c.head_dim, num_blocks=users * blocks_each + 1, device=DEVICE)
    for layer in range(c.layers):
        pool.k[layer].normal_()
        pool.v[layer].normal_()
    tables = [allocator.allocate(blocks_each) for _ in range(users)]
    return pool, tables, [length] * users, [100] * users


def plain_ms(model, pool, tables, lengths, tokens):
    """Average time of one model.decode call, with no stopwatches inside."""
    for _ in range(3):
        model.decode(tokens, tables, lengths, pool)
    gpu_wait()
    start = time.perf_counter()
    for _ in range(REPEATS):
        model.decode(tokens, tables, lengths, pool)
    gpu_wait()
    return (time.perf_counter() - start) / REPEATS * 1000


def split_ms(model, pool, tables, lengths, tokens):
    """The same decode step as model.decode, with the GPU drained and a stopwatch read around every part."""
    totals = dict.fromkeys(("setup", "norm+qkv+rope", "write", "gather", "attention", "o_proj+mlp"), 0.0)
    for _ in range(REPEATS):
        gpu_wait()
        last = time.perf_counter()

        def lap(part):
            nonlocal last
            gpu_wait()
            now = time.perf_counter()
            totals[part] += now - last
            last = now

        c, bs = model.config, pool.block_size
        used = [n // bs + 1 for n in lengths]
        width = max(used)
        index = torch.tensor([t[:u] + [0] * (width - u) for t, u in zip(tables, used)], device=DEVICE)
        blocks = torch.tensor([t[n // bs] for t, n in zip(tables, lengths)], device=DEVICE)
        slots = torch.tensor([n % bs for n in lengths], device=DEVICE)
        positions = torch.tensor(lengths, device=DEVICE)
        mask = (torch.arange(width * bs, device=DEVICE)[None] <= positions[:, None])[:, None]
        cos, sin = rope_tables(positions[:, None], c.head_dim, c.rope_theta)
        cos, sin = cos[:, None], sin[:, None]
        x = F.embedding(torch.tensor(tokens, device=DEVICE)[:, None], model.embed)
        lap("setup")
        for i, layer in enumerate(model.layers):
            q, k, v = model.qkv(rms_norm(x, layer["attn_norm"], c.eps), layer, cos, sin)
            lap("norm+qkv+rope")
            pool.write(i, blocks, slots, k[:, :, 0], v[:, :, 0])
            lap("write")
            k_all = pool.k[i][index].flatten(1, 2).transpose(1, 2)
            v_all = pool.v[i][index].flatten(1, 2).transpose(1, 2)
            lap("gather")
            out = paged_attention(q, k_all, v_all, mask)
            lap("attention")
            x = x + F.linear(out.transpose(1, 2).flatten(2), layer["o"])
            x = x + model.mlp(rms_norm(x, layer["mlp_norm"], c.eps), layer)
            lap("o_proj+mlp")
    return {part: seconds / REPEATS * 1000 for part, seconds in totals.items()}


def main():
    """Print, for each batch size and context length, the plain decode time and where the time goes inside it."""
    model = SmolLM2.from_pretrained(device=DEVICE)
    print(f"device: {DEVICE}, times in ms per decode step (30 layers)")
    header = None
    with torch.no_grad():
        for users, length in CASES:
            pool, tables, lengths, tokens = setup(model, users, length)
            plain = plain_ms(model, pool, tables, lengths, tokens)
            parts = split_ms(model, pool, tables, lengths, tokens)
            if header is None:
                header = "users  context   plain  " + "  ".join(f"{p:>13}" for p in parts)
                print(header)
            print(f"{users:>5}  {length:>7}  {plain:>6.1f}  " + "  ".join(f"{ms:>13.2f}" for ms in parts.values()))
            del pool


if __name__ == "__main__":
    main()
