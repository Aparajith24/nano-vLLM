from collections import deque

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

