import sys

import torch

from blocks import BlockAllocator, KVPool

LAYERS, KV_HEADS, HEAD_DIM, NUM_BLOCKS = 2, 3, 4, 12
LENGTHS = (15, 16, 17, 32, 33)


def known_values(user, layer, start, count, device):
    """K values that say exactly which user, layer, token, head and dim they belong to; V is the negative."""
    token = torch.arange(start, start + count, device=device).float()[:, None, None]
    head = torch.arange(KV_HEADS, device=device).float()[None, :, None]
    dim = torch.arange(HEAD_DIM, device=device).float()[None, None, :]
    k = user * 10000 + layer * 1000 + token + head * 0.1 + dim * 0.01
    return k, -k


def run(device, length):
    """Write two users token by token and a third all at once, then read every user back through their table."""
    allocator = BlockAllocator(num_blocks=NUM_BLOCKS)
    pool = KVPool(LAYERS, KV_HEADS, HEAD_DIM, num_blocks=NUM_BLOCKS, device=device)
    for layer in range(LAYERS):
        pool.k[layer].copy_(torch.randn_like(pool.k[layer]) * 1e4)
        pool.v[layer].copy_(torch.randn_like(pool.v[layer]) * 1e4)
    tables = {1: [], 2: [], 3: []}
    for n in range(length):
        for user in (1, 2):
            allocator.grow(tables[user], n + 1)
            blocks, slots = pool.locations(tables[user], n, 1)
            for layer in range(LAYERS):
                pool.write(layer, blocks, slots, *known_values(user, layer, n, 1, device))
    allocator.grow(tables[3], length)
    blocks, slots = pool.locations(tables[3], 0, length)
    for layer in range(LAYERS):
        pool.write(layer, blocks, slots, *known_values(3, layer, 0, length, device))
    allocator.check()
    for user, table in tables.items():
        for layer in range(LAYERS):
            k, v = pool.read(layer, table, length)
            expected_k, expected_v = known_values(user, layer, 0, length, device)
            assert torch.equal(k, expected_k) and torch.equal(v, expected_v), f"user {user} layer {layer} wrong"
    needs_new = len(tables[1]) * pool.block_size == length
    print(f"  length {length:>2}: tables {tables[1]} {tables[2]} {tables[3]}  read back exactly  "
          f"next token needs a new block: {needs_new}")


def main(devices):
    """Run the write-and-read check at every length on every device."""
    for device in devices:
        print(f"device: {device}")
        for length in LENGTHS:
            run(device, length)
    print("all write path checks passed")


if __name__ == "__main__":
    main(sys.argv[1:] or ["cpu", "mps"])
