from typing import List, Tuple, Optional, Sequence, Dict
import numpy as np
import math
import random


def verify_progressive(seq: List[int], n: int) -> None:
    """Verifies the PMJ 1-per-cell stratification at prefix lengths 4^k."""
    levels = int(math.log2(n))
    assert 2 ** levels == n

    pts = [(i % n, i // n) for i in seq]

    for k in range(1, levels + 1):
        prefix = pts[: 4 ** k]
        size = n // (2 ** k)  # block size at level k
        counts = [[0 for _ in range(2 ** k)] for __ in range(2 ** k)]
        for (x, y) in prefix:
            bx = x // size
            by = y // size
            counts[by][bx] += 1
        bad = [(by, bx, counts[by][bx])
               for by in range(2 ** k)
               for bx in range(2 ** k)
               if counts[by][bx] != 1]
        assert not bad, f"Level {k} failed blocks: {bad[:5]} (showing up to 5)"


def cumulative_checkpoints(num_transfer_tokens: Sequence[int]) -> List[int]:
    s = 0
    out = []
    for v in num_transfer_tokens:
        s += int(v)
        out.append(s)
    return out


def pmj_blue_noise_ordering(
    n: int = 64,
    seed: Optional[int] = None,
    shuffle_blocks: bool = True,
    candidates_per_block: int = 8,     # cell-level BN pressure: 4–16 is reasonable
    orders_per_level: int = 6,         # batch-level BN pressure: 4–8 recommended
    checkpoints: Optional[Sequence[int]] = None,
    # advanced: reduce orders in late levels to control runtime
    orders_schedule: Optional[Dict[int, int]] = None,  # map: level_idx(1..log2(n)) -> orders_per_level
) -> Tuple[List[int], np.ndarray]:
    """
    PMJ ordering with cell-level and batch-level blue-noise optimization against
    a given set of cumulative 'checkpoints' (prefix sizes). Returns (row-major indices, points_2xN).

    - 'checkpoints' should be strictly increasing and end at n*n (e.g., your cumsum).
    - Preserves PMJ progressive property at 4^k prefixes.
    """
    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError("n must be a positive power of two (e.g., 64)")
    N = n * n
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    # Occupancy grid and list of chosen points (ints)
    occupied = np.zeros((n, n), dtype=bool)
    chosen_xy: List[Tuple[int, int]] = []  # order of placements

    # Blocks are tuples: (x0, y0, size)
    blocks: List[Tuple[int, int, int]] = [(0, 0, n)]

    # Prepare checkpoints
    if checkpoints is None:
        checkpoints = [N]
    checkpoints = list(checkpoints)
    assert len(checkpoints) >= 1 and checkpoints[-1] == N
    # next checkpoint index we must satisfy
    ck_idx = 0
    # current prefix length
    cur_m = 0

    def np_points_array() -> np.ndarray:
        if not chosen_xy:
            return np.empty((0, 2), dtype=float)
        arr = np.array(chosen_xy, dtype=float)
        return arr  # integer coords; scaling not needed for maximization

    def place_point(x: int, y: int):
        nonlocal cur_m
        occupied[y, x] = True
        chosen_xy.append((x, y))
        cur_m += 1

    def best_candidate_in_block(x0: int, y0: int, size: int) -> Tuple[int, int]:
        """
        Return the (x,y) in this sub-block that maximizes min-distance to already chosen points.
        If no points yet, just random.
        """
        if cur_m == 0:
            x = rng.randrange(x0, x0 + size)
            y = rng.randrange(y0, y0 + size)
            return x, y

        existing = np_points_array()  # (m,2)
        # Draw candidate cells
        cand_x = np_rng.integers(low=x0, high=x0 + size, size=candidates_per_block)
        cand_y = np_rng.integers(low=y0, high=y0 + size, size=candidates_per_block)
        # Ensure uniqueness within block to avoid wasted evals
        cands = np.unique(np.stack([cand_x, cand_y], axis=1), axis=0)
        # Safety: ensure at least one
        if cands.shape[0] == 0:
            cands = np.array([[rng.randrange(x0, x0 + size), rng.randrange(y0, y0 + size)]], dtype=int)

        # All blocks we call this on are empty (no previous sample in the block),
        # but we still ensure we don't re-use a cell (global occupied check)
        # If a rare collision occurs (shouldn't), redraw a few times.
        for _ in range(4):
            mask_occ = occupied[cands[:, 1], cands[:, 0]]
            if not np.any(mask_occ):
                break
            cands = cands[~mask_occ]
            if cands.shape[0] == 0:
                # redraw
                cand_x = np_rng.integers(low=x0, high=x0 + size, size=candidates_per_block)
                cand_y = np_rng.integers(low=y0, high=y0 + size, size=candidates_per_block)
                cands = np.unique(np.stack([cand_x, cand_y], axis=1), axis=0)

        if cands.shape[0] == 1 or cur_m == 0:
            return int(cands[0, 0]), int(cands[0, 1])

        # Compute min distance to existing for each candidate
        # distances: ||c - p|| for p in existing
        # We'll compute squared distances to save sqrt, but we need true distances later for sums.
        # For selection (max min-dist), squared is fine.
        ex = existing[:, 0]
        ey = existing[:, 1]
        best_idx = 0
        best_min_d2 = -1.0
        for i, (cx, cy) in enumerate(cands):
            dx = ex - cx
            dy = ey - cy
            d2 = dx * dx + dy * dy
            min_d2 = float(np.min(d2))
            if min_d2 > best_min_d2:
                best_min_d2 = min_d2
                best_idx = i
        bx, by = cands[best_idx]
        return int(bx), int(by)

    # For batch-level order evaluation, we need the running sum of pairwise distances S(m)
    # mean pairwise at m points: M(m) = 2 * S(m) / (m*(m-1))
    def eval_order_incremental_objective(
        start_points: np.ndarray,            # (m0,2) points before this level
        candidate_points: np.ndarray,        # (L,2) points for this level (order is a permutation)
        checkpoints_in_level: List[int],     # absolute m-values (prefix sizes) within this level's range
    ) -> float:
        """
        Simulate inserting candidate_points in order, compute the *sum of mean pairwise distances*
        at the required checkpoints_in_level. Return that sum for comparison across different orders.
        """
        # Running S and current points array
        pts = start_points
        m = pts.shape[0]
        # Initialize S(m) from scratch ONLY once per order (for speed, we can cache per level,
        # but simplicity: compute here).
        if m <= 1:
            S = 0.0
        else:
            # S = sum_{i<j} ||p_i - p_j||
            # Vectorized but O(m^2); m is small in early levels and larger later.
            # To reduce cost, we only recompute once per order.
            diffs = pts[None, :, :] - pts[:, None, :]
            dists = np.sqrt(np.sum(diffs * diffs, axis=2))
            S = float(np.triu(dists, k=1).sum())

        # We’ll walk through candidate_points, update S incrementally:
        # When adding point q, new S += sum(||q - p_i||) over all previous points
        checkpoints_sorted = sorted(checkpoints_in_level)
        ck_ptr = 0
        obj_sum = 0.0
        # pre-allocate growth buffers
        for q in candidate_points:
            if pts.size == 0:
                inc = 0.0
            else:
                diffs = pts - q  # (m,2)
                inc = float(np.sqrt(np.sum(diffs * diffs, axis=1)).sum())
            S += inc
            # append
            if pts.size == 0:
                pts = q.reshape(1, 2)
            else:
                pts = np.vstack([pts, q])
            m += 1

            # consume checkpoints reached
            while ck_ptr < len(checkpoints_sorted) and m == checkpoints_sorted[ck_ptr]:
                if m >= 2:
                    mean_pair = 2.0 * S / (m * (m - 1))
                else:
                    mean_pair = 0.0
                obj_sum += mean_pair
                ck_ptr += 1

            # early break if we passed all checkpoints in this level
            if ck_ptr >= len(checkpoints_sorted):
                # We can stop simulating if we like, but the caller may reuse pts; continue is fine.
                pass

        return obj_sum

    def sum_orders_at_level(
        new_points: List[Tuple[int, int]],
        level_idx: int,
        ck_abs: List[int],  # absolute checkpoints that lie within [cur_m+1, cur_m+len(new_points)]
    ) -> List[Tuple[float, List[int]]]:
        """
        Given the set of candidate placements for this level (unordered),
        try multiple orders, return list of (score, order_indices) sorted descending by score.
        """
        if not new_points:
            return [(0.0, [])]

        # The number of trial orders for this level
        local_orders = orders_per_level
        if orders_schedule and level_idx in orders_schedule:
            local_orders = max(1, int(orders_schedule[level_idx]))

        L = len(new_points)
        start_pts = np_points_array()

        # Pre-build one greedy order:
        # At each step, pick point with largest sum of distances to current set.
        # Greedy is O(L^2 * m) worst-case; acceptable for moderate L (it can be heavy for 3072).
        # We’ll include it when L is not too large; otherwise skip greedy to save time.
        orders: List[List[int]] = []
        idxs = list(range(L))
        if L <= 1024:  # heuristic guard
            remaining = new_points[:]  # shallow copy of tuples
            cur = start_pts.copy()
            order_greedy: List[int] = []
            while remaining:
                # compute sum-dist for each candidate to current set
                if cur.shape[0] == 0:
                    # arbitrary first
                    best_j = 0
                    inc_best = 0.0
                else:
                    ex = cur
                    sums = []
                    for j, (x, y) in enumerate(remaining):
                        diffs = ex - np.array([[x, y]], dtype=float)
                        sums.append(float(np.sqrt(np.sum(diffs * diffs, axis=1)).sum()))
                    best_j = int(np.argmax(sums))
                # commit
                order_greedy.append(idxs[best_j])
                # update
                point = np.array([[remaining[best_j][0], remaining[best_j][1]]], dtype=float)
                cur = point if cur.size == 0 else np.vstack([cur, point])
                del remaining[best_j]
                del idxs[best_j]
            orders.append(order_greedy)
            # reset idxs for subsequent random orders
            idxs = list(range(L))
        else:
            # Skip greedy when too large; random orders will be used.
            pass

        # Random shuffles
        n_rand = max(0, local_orders - len(orders))
        for _ in range(n_rand):
            r = idxs[:]
            rng.shuffle(r)
            orders.append(r)

        # Evaluate each order against the checkpoints in this level
        results: List[Tuple[float, List[int]]] = []
        cand_arr = np.array(new_points, dtype=float)  # (L,2)
        for ord_idx, order in enumerate(orders):
            arr = cand_arr[order]  # reorder
            score = eval_order_incremental_objective(start_pts, arr, ck_abs)
            results.append((score, order))

        # Sort by descending score
        results.sort(key=lambda t: t[0], reverse=True)
        return results

    # Iterate PMJ levels
    size = n
    level_idx = 0
    while size > 1:
        level_idx += 1
        half = size // 2
        # Collect child blocks, keep visitation order random if desired
        children: List[Tuple[int, int, int]] = []
        for (x0, y0, s) in blocks:
            assert s == size
            children.extend([
                (x0, y0, half),                # NW
                (x0 + half, y0, half),         # NE
                (x0, y0 + half, half),         # SW
                (x0 + half, y0 + half, half),  # SE
            ])
        if shuffle_blocks:
            rng.shuffle(children)

        # Determine which children need a sample (those empty)
        # (We keep one per child to preserve PMJ structure)
        to_fill: List[Tuple[int, int, int]] = []
        for (x0, y0, s) in children:
            # check if any occupied inside this child
            sub = occupied[y0:y0 + s, x0:x0 + s]
            if not np.any(sub):
                to_fill.append((x0, y0, s))

        # For each required child block, pick a best-candidate point (cell-level BN)
        new_points: List[Tuple[int, int]] = []
        for (x0, y0, s) in to_fill:
            bx, by = best_candidate_in_block(x0, y0, s)
            new_points.append((bx, by))

        # Figure absolute checkpoints that fall in this level’s insertion window
        L = len(new_points)
        level_range = (cur_m + 1, cur_m + L)  # inclusive endpoints in 1-based counting
        ck_abs = [c for c in checkpoints if level_range[0] <= c <= level_range[1]]

        # If there are checkpoints to evaluate in this level, choose the best order
        # (batch-level BN). Otherwise, any order is fine; still prefer greedy/random mix.
        if L > 0:
            candidate_orders = sum_orders_at_level(new_points, level_idx, ck_abs)
            best_score, best_order = candidate_orders[0] if candidate_orders else (0.0, list(range(L)))
            # Commit points in that order
            for idx in best_order:
                x, y = new_points[idx]
                place_point(x, y)
                # Advance checkpoints if we just hit them
                while ck_idx < len(checkpoints) and cur_m == checkpoints[ck_idx]:
                    ck_idx += 1

            # In case no candidate_orders returned (shouldn’t), fallback deterministic
            if not candidate_orders:
                for (x, y) in new_points:
                    place_point(x, y)
                    while ck_idx < len(checkpoints) and cur_m == checkpoints[ck_idx]:
                        ck_idx += 1

        # Prepare next level
        blocks = children
        size = half

    # Finally, any remaining empty 1x1 cells—just append them; no checkpoints remain beyond N.
    # (All leaf cells are 1x1 with either used or unused.)
    for y in range(n):
        for x in range(n):
            if not occupied[y, x]:
                place_point(x, y)

    # Build outputs
    seq_linear: List[int] = [y * n + x for (x, y) in chosen_xy]
    pts2 = np.array(chosen_xy, dtype=float).T  # shape (2, N)
    # Sanity: verify PMJ stratification
    verify_progressive(seq_linear, n)
    return seq_linear, pts2


# --- An objective helper mirroring your get_avg() with pdist ---
def get_avg_with_pdist(ordering_xy_2xN: np.ndarray, num_transfer_tokens: Sequence[int]) -> List[float]:
    """
    ordering_xy_2xN: shape (2, N) ints
    """
    from scipy.spatial.distance import pdist  # if SciPy is unavailable, we can reimplement
    points = ordering_xy_2xN.T
    avg_distances = []
    x = 0
    for step in num_transfer_tokens:
        subset = points[:x + step]
        x += step
        if subset.shape[0] <= 1:
            avg_distances.append(0.0)
        else:
            dists = pdist(subset, metric='euclidean')
            avg_distances.append(float(np.mean(dists)))
    return avg_distances


if __name__ == "__main__":
    # Example usage
    n = 64
    num_transfer_tokens = [43, 47, 50, 56, 60, 68, 75, 83, 94, 107, 122, 141, 164, 194,
                           232, 285, 355, 458, 609, 853]
    checkpoints = cumulative_checkpoints(num_transfer_tokens)  # ends at 4096

    # Baseline: your original PMJ (for comparison)
    # (You can plug your function here to compute avg distances.)

    # Optimized PMJ with two-level blue-noise
    seq_opt, pts2_opt = pmj_blue_noise_ordering(
        n=n,
        seed=42,
        shuffle_blocks=True,
        candidates_per_block=12,  # increase for stronger cell-level BN
        orders_per_level=6,       # try 4–8; reduce if runtime is high
        checkpoints=checkpoints,
        # Optional: fewer orders at the last level to speed up:
        orders_schedule={6: 3}    # level_idx=6 (the 3072-point level): try only 3 orders
    )

    # Compute your objective values
    avgs_opt = get_avg_with_pdist(pts2_opt, num_transfer_tokens)
    print("Optimized avg pairwise distances per batch:", avgs_opt)
    print("prefix sizes ok at 4^k for k=1..6")

