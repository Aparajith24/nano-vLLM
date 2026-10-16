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
    print("all allocator checks passed")


if __name__ == "__main__":
    main()
