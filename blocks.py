from collections import deque

import torch

BLOCK_SIZE = 16
DEFAULT_NUM_BLOCKS = 1920
DUMMY_BLOCK = 0


class PoolFull(Exception):
    """Raised when a request needs more blocks than are free; nothing is handed out."""


class BlockAllocator:
    """Hands out block numbers from a fixed pool, counts how many users share each one, and remembers full prompt blocks for reuse."""

    def __init__(self, num_blocks=DEFAULT_NUM_BLOCKS, block_size=BLOCK_SIZE):
        self.num_blocks, self.block_size = num_blocks, block_size
        self.free_blocks = deque(range(1, num_blocks))
        self.refs, self.cached, self.hash_of = {}, {}, {}

    @property
    def used(self):
        """The blocks that at least one user holds."""
        return set(self.refs)

    def num_free(self):
        """How many blocks can still be handed out."""
        return len(self.free_blocks)

    def blocks_for(self, tokens):
        """How many blocks it takes to hold this many tokens."""
        return -(-tokens // self.block_size)

    def take(self, block):
        """Hand out one block from the free list, forgetting any prompt it used to hold."""
        old_hash = self.hash_of.pop(block, None)
        if old_hash is not None:
            del self.cached[old_hash]
        self.refs[block] = 1
        return block

    def allocate(self, count):
        """Take this many free blocks, or raise PoolFull and take none."""
        if count > len(self.free_blocks):
            raise PoolFull(f"need {count} blocks, only {len(self.free_blocks)} free")
        return [self.take(self.free_blocks.popleft()) for _ in range(count)]

    def free(self, blocks):
        """Drop one user's hold on these blocks; a block nobody holds goes back to the free list but keeps its prompt for reuse."""
        if len(set(blocks)) != len(blocks) or not self.used.issuperset(blocks):
            raise ValueError(f"cannot free {blocks}: a block is not in use or listed twice")
        for block in blocks:
            self.refs[block] -= 1
            if self.refs[block] == 0:
                del self.refs[block]
                self.free_blocks.append(block)

    def grow(self, table, tokens):
        """Add blocks to a block table until it has room for this many tokens."""
        missing = self.blocks_for(tokens) - len(table)
        if missing > 0:
            table.extend(self.allocate(missing))
        return table

    def prefix_hashes(self, ids):
        """A fingerprint for every full block of ids; each one also covers every token before that block."""
        hashes, previous = [], None
        for start in range(0, len(ids) - self.block_size + 1, self.block_size):
            previous = hash((previous, tuple(ids[start:start + self.block_size])))
            hashes.append(previous)
        return hashes

    def allocate_for(self, ids, total_tokens, use_cache=True):
        """A table with room for total_tokens: cached blocks for the longest known prefix of ids (never its last token), new blocks for the rest; returns (table, cached tokens) or raises PoolFull and changes nothing."""
        hits = []
        if use_cache:
            for block_hash in self.prefix_hashes(ids[:-1]):
                if block_hash not in self.cached:
                    break
                hits.append(self.cached[block_hash])
        revived = [block for block in hits if block not in self.refs]
        new = self.blocks_for(total_tokens) - len(hits)
        if new > len(self.free_blocks) - len(revived):
            raise PoolFull(f"need {new} new blocks, only {len(self.free_blocks) - len(revived)} free")
        for block in hits:
            if block in self.refs:
                self.refs[block] += 1
            else:
                self.free_blocks.remove(block)
                self.refs[block] = 1
        return hits + self.allocate(new), len(hits) * self.block_size

    def register(self, table, ids):
        """Remember the full blocks of a prompt that has just been written, so later users with the same prefix can share them."""
        for block, block_hash in zip(table, self.prefix_hashes(ids)):
            if block_hash not in self.cached and block not in self.hash_of:
                self.cached[block_hash], self.hash_of[block] = block, block_hash

    def check(self):
        """Assert that every real block is either free or held, never both and never lost, and that the prompt index is consistent."""
        free = set(self.free_blocks)
        assert len(free) == len(self.free_blocks), "a block is free twice"
        assert not free & self.used, "a block is both free and used"
        assert free | self.used == set(range(1, self.num_blocks)), "a block went missing"
        assert DUMMY_BLOCK not in free | self.used, "the dummy block was handed out"
        assert all(count > 0 for count in self.refs.values()), "a held block has no holder"
        assert {b: h for h, b in self.cached.items()} == self.hash_of, "the prompt index is inconsistent"


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
