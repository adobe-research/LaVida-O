from typing import List, Tuple, Optional
import random


def stratified_random(n: int = 64, seed: Optional[int] = None, shuffle_blocks: bool = True) -> List[int]:
    """
    Progressive Multi‑Jittered (PMJ) ordering over an n×n integer grid, n must be a power of two.

    The algorithm recursively subdivides the full grid into 2×2 blocks. At each level, it ensures
    every sub‑block contains exactly one sample by placing a new integer‑grid point uniformly at
    random in each sub‑block that doesn't already contain a sample from a previous level. The
    resulting sequence is progressive: the first 4^k samples are 1‑per‑cell stratified over a
    (n/2^k)×(n/2^k) tiling of the domain.

    Returns
    -------
    List[int]
        Row‑major linear indices y*n + x for x,y in [0, n).
    """
    # Validate power of two
    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError("n must be a positive power of two (e.g., 64)")

    rng = random.Random(seed)

    # Occupancy grid: False means empty, True means already sampled
    occupied = [[False] * n for _ in range(n)]

    # The progressive sequence (row‑major linear indices)
    seq: List[int] = []

    # A block is represented as (x0, y0, size)
    blocks: List[Tuple[int, int, int]] = [(0, 0, n)]

    def block_has_sample(x0: int, y0: int, size: int) -> bool:
        for yy in range(y0, y0 + size):
            row = occupied[yy]
            for xx in range(x0, x0 + size):
                if row[xx]:
                    return True
        return False

    def place_random_in_block(x0: int, y0: int, size: int):
        # Because we only call this on blocks known to be empty, a single draw suffices.
        x = rng.randrange(x0, x0 + size)
        y = rng.randrange(y0, y0 + size)
        # Safety: loop until we hit an empty cell (should be first try for empty block)
        attempts = 0
        while occupied[y][x]:
            x = rng.randrange(x0, x0 + size)
            y = rng.randrange(y0, y0 + size)
            attempts += 1
            if attempts > 10000:
                raise RuntimeError("Too many attempts to place a sample; logic error?")
        occupied[y][x] = True
        seq.append(y * n + x)

    # Iterate levels until block size == 1
    size = n
    while size > 1:
        # Subdivide each block into 4 children
        half = size // 2
        children: List[Tuple[int, int, int]] = []
        for (x0, y0, s) in blocks:
            assert s == size
            children.extend([
                (x0, y0, half),                # NW
                (x0 + half, y0, half),         # NE
                (x0, y0 + half, half),         # SW
                (x0 + half, y0 + half, half),  # SE
            ])
        # Optionally randomize visitation order to reduce directional bias in sequence order
        if shuffle_blocks:
            rng.shuffle(children)
        # For each child, if empty, place a random sample inside it
        for (x0, y0, s) in children:
            if not block_has_sample(x0, y0, s):
                place_random_in_block(x0, y0, s)
        # Next level
        blocks = children
        size = half

    # At this point, every 1×1 cell is a block; any still‑empty cells need to be appended
    # (these are exactly those not yet selected at previous levels).
    # To preserve the progressive property, all remaining cells are appended in a random order.
    # (Any order works because they are all 1×1; using random makes ties less structured.)
    remaining: List[int] = []
    for y in range(n):
        for x in range(n):
            if not occupied[y][x]:
                remaining.append(y * n + x)
    rng.shuffle(remaining)
    seq.extend(remaining)

    assert len(seq) == n * n, (len(seq), n * n)

    return seq


def verify_progressive(seq: List[int], n: int) -> None:
    """Debug helper: verifies the PMJ 1‑per‑cell stratification at prefix lengths 4^k."""
    import math
    levels = int(math.log2(n))
    assert 2 ** levels == n

    # Convert linear indices to (x, y)
    pts = [(i % n, i // n) for i in seq]

    for k in range(1, levels + 1):
        prefix = pts[: 4 ** k]
        size = n // (2 ** k)  # block size at level k
        # Count how many points fall into each block
        counts = [[0 for _ in range(2 ** k)] for __ in range(2 ** k)]
        for (x, y) in prefix:
            bx = x // size
            by = y // size
            counts[by][bx] += 1
        # Every block must contain exactly one point
        bad = [(by, bx, counts[by][bx])
               for by in range(2 ** k)
               for bx in range(2 ** k)
               if counts[by][bx] != 1]
        assert not bad, f"Level {k} failed blocks: {bad[:5]} (showing up to 5)"


if __name__ == "__main__":
    n = 64
    seq = stratified_random(n=n, seed=42, shuffle_blocks=True)
    verify_progressive(seq, n)
    # Show a few samples and some prefix sizes
    print(seq[:10])
    print("prefix sizes ok at 4^k for k=1..6")
