from collections import deque

import torch

BLOCK_SIZE = 16
DEFAULT_NUM_BLOCKS = 1920
DUMMY_BLOCK = 0


class PoolFull(Exception):
    """Raised when a request needs more blocks than are free; nothing is handed out."""


class BlockAllocator:
    """Hands out block numbers from a fixed pool and takes them back; block 0 is a dummy that is never handed out."""

    def __init__(self, num_blocks=DEFAULT_NUM_BLOCKS, block_size=BLOCK_SIZE):
        self.num_blocks, self.block_size = num_blocks, block_size
        self.free_blocks = deque(range(1, num_blocks))
        self.used = set()

    def num_free(self):
        """How many blocks can still be handed out."""
        return len(self.free_blocks)

    def blocks_for(self, tokens):
        """How many blocks it takes to hold this many tokens."""
        return -(-tokens // self.block_size)

    def allocate(self, count):
        """Take this many free blocks, or raise PoolFull and take none."""
        if count > len(self.free_blocks):
            raise PoolFull(f"need {count} blocks, only {len(self.free_blocks)} free")
        blocks = [self.free_blocks.popleft() for _ in range(count)]
        self.used.update(blocks)
        return blocks

    def free(self, blocks):
        """Give blocks back to the pool, or raise and give back none if any block is not in use or listed twice."""
        if len(set(blocks)) != len(blocks) or not self.used.issuperset(blocks):
            raise ValueError(f"cannot free {blocks}: a block is not in use or listed twice")
        self.used.difference_update(blocks)
        self.free_blocks.extend(blocks)

    def grow(self, table, tokens):
        """Add blocks to a block table until it has room for this many tokens."""
        missing = self.blocks_for(tokens) - len(table)
        if missing > 0:
            table.extend(self.allocate(missing))
        return table

    def check(self):
        """Assert that every real block is either free or used, never both and never lost."""
        free = set(self.free_blocks)
        assert len(free) == len(self.free_blocks), "a block is free twice"
        assert not free & self.used, "a block is both free and used"
        assert free | self.used == set(range(1, self.num_blocks)), "a block went missing"
        assert DUMMY_BLOCK not in free | self.used, "the dummy block was handed out"


class KVPool:
    """The real memory behind the blocks: one K and one V tensor per layer, shaped [blocks, block_size, kv_heads, head_dim]."""

    def __init__(self, layers, kv_heads, head_dim, num_blocks=DEFAULT_NUM_BLOCKS, block_size=BLOCK_SIZE,
                 device="cpu", dtype=torch.float32):
        shape = (num_blocks, block_size, kv_heads, head_dim)
        self.block_size, self.device = block_size, device
        self.k = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(layers)]
        self.v = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(layers)]

    def locations(self, table, start, count):
        """The (block, slot) of each token from position start to start + count - 1, as two tensors."""
        positions = range(start, start + count)
        blocks = [table[n // self.block_size] for n in positions]
        slots = [n % self.block_size for n in positions]
        return torch.tensor(blocks, device=self.device), torch.tensor(slots, device=self.device)

    def write(self, layer, blocks, slots, k, v):
        """Store K and V of shape [tokens, kv_heads, head_dim] at the given blocks and slots of one layer."""
        self.k[layer][blocks, slots] = k
        self.v[layer][blocks, slots] = v

    def read(self, layer, table, length):
        """One user's first length tokens of K and V for one layer, gathered through their block table."""
        index = torch.tensor(table, device=self.device)
        k = self.k[layer][index].flatten(0, 1)[:length]
        v = self.v[layer][index].flatten(0, 1)[:length]
        return k, v
