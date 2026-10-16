from blocks import BlockAllocator, PoolFull


def step(pool, label, tables, expected_tables, expected_free):
    """Print the pool after one step and assert the tables and free list are what the paper drawing says."""
    pool.check()
    print(f"{label:<34} " + "  ".join(f"{name}: {table}" for name, table in tables.items())
          + f"  free: {list(pool.free_blocks)}")
    assert tables == expected_tables, f"tables {tables} != {expected_tables}"
    assert list(pool.free_blocks) == expected_free, f"free {list(pool.free_blocks)} != {expected_free}"


def main():
    """Replay the paper toy: 6 blocks of 4 tokens, block 0 reserved, users arriving, growing and leaving."""
    pool = BlockAllocator(num_blocks=6, block_size=4)
    tables = {}
    step(pool, "start", tables, {}, [1, 2, 3, 4, 5])

    tables["A"] = pool.grow([], 6)
    step(pool, "A arrives with 6 tokens", tables, {"A": [1, 2]}, [3, 4, 5])
    for tokens in (7, 8):
        pool.grow(tables["A"], tokens)
        step(pool, f"A grows to {tokens}", tables, {"A": [1, 2]}, [3, 4, 5])
    pool.grow(tables["A"], 9)
    step(pool, "A grows to 9", tables, {"A": [1, 2, 3]}, [4, 5])

    tables["B"] = pool.grow([], 2)
    step(pool, "B arrives with 2 tokens", tables, {"A": [1, 2, 3], "B": [4]}, [5])

    try:
        pool.grow([], 9)
        raise AssertionError("C should not fit")
    except PoolFull as error:
        print(f"{'C asks for 9 tokens (3 blocks)':<34} refused: {error}")
    step(pool, "after the refusal", tables, {"A": [1, 2, 3], "B": [4]}, [5])

    pool.free(tables.pop("A"))
    step(pool, "A leaves", tables, {"B": [4]}, [5, 1, 2, 3])
    pool.grow(tables["B"], 5)
    step(pool, "B grows to 5", tables, {"B": [4, 5]}, [1, 2, 3])
    tables["C"] = pool.grow([], 9)
    step(pool, "C arrives with 9 tokens", tables, {"B": [4, 5], "C": [1, 2, 3]}, [])

    for label, bad in (("free the dummy block", [0]), ("free a block twice in one call", [4, 4])):
        try:
            pool.free(bad)
            raise AssertionError(f"{label} should fail")
        except ValueError as error:
            print(f"{label:<34} refused: {error}")
    step(pool, "after the bad frees", tables, {"B": [4, 5], "C": [1, 2, 3]}, [])

    blocks_of_b = tables.pop("B")
    pool.free(blocks_of_b)
    try:
        pool.free(blocks_of_b)
        raise AssertionError("freeing B twice should fail")
    except ValueError as error:
        print(f"{'B leaves, then is freed again':<34} refused: {error}")
    pool.free(tables.pop("C"))
    step(pool, "everyone leaves", tables, {}, [4, 5, 1, 2, 3])
    prefix_cases()
    print("all allocator checks passed")


def prefix_cases():
    """Share, revive and evict cached prompt blocks with a toy pool of 8 blocks of 4 tokens."""
    pool = BlockAllocator(num_blocks=8, block_size=4)
    shared = [11, 12, 13, 14, 15, 16, 17, 18]
    a_ids, b_ids = shared + [21, 22], shared + [31]
    a, cached = pool.allocate_for(a_ids, 12)
    assert (a, cached) == ([1, 2, 3], 0)
    pool.register(a, a_ids)
    b, cached = pool.allocate_for(b_ids, 10)
    assert (b, cached) == ([1, 2, 4], 8), (b, cached)
    pool.check()
    print(f"{'B shares A first 8 tokens':<34} A: {a}  B: {b}  holders: {pool.refs}")
    pool.free(a)
    pool.free(b)
    pool.check()
    assert pool.refs == {} and pool.cached
    print(f"{'A and B leave':<34} free: {list(pool.free_blocks)}  still cached: {sorted(pool.cached.values())}")
    c, cached = pool.allocate_for(shared + [41, 42, 43], 12)
    assert (c[:2], cached) == ([1, 2], 8), (c, cached)
    pool.check()
    print(f"{'C revives the cached blocks':<34} C: {c}  free: {list(pool.free_blocks)}")
    whole, cached = pool.allocate_for(shared, 8)
    assert cached == 4, "the last token of a prompt is never served from the cache"
    pool.free(whole)
    print(f"{'same 8-token prompt again':<34} cached tokens: {cached} (block 2 holds the last token, recomputed)")
    pool.free(c)
    filler = pool.allocate(pool.num_free())
    pool.check()
    assert not pool.cached, "taking every free block must evict every cached prompt"
    pool.free(filler)
    d, cached = pool.allocate_for(shared + [51], 9)
    assert cached == 0
    pool.check()
    print(f"{'pool fills up, then D arrives':<34} cached tokens: {cached} (evicted)")


if __name__ == "__main__":
    main()
