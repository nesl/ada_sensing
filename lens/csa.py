# csa.py
from __future__ import annotations
from typing import Any, Dict, List, Sequence
import random
import math

def csa1_random(all_ids: Sequence[int], k: int, rng: random.Random) -> List[int]:
    k = min(k, len(all_ids))
    return rng.sample(list(all_ids), k)

def csa2_grid_random(
    options: Sequence[Dict[str, Any]],
    k: int,
    rng: random.Random,
    grid_bins_per_dim: int = 3,
    dims: Sequence[str] = ("iso", "ss", "ap"),
) -> List[int]:
    """
    Grid-based random selection over a discrete cube.
    Each option should have meta values for dims, either numeric or orderable tokens.
    We bucket each dim into bins based on rank in sorted unique values.
    """
    if k >= len(options):
        return [o["option_id"] for o in options]

    # Build rank maps per dim
    unique_vals = {d: sorted({o["meta"].get(d) for o in options}) for d in dims}
    rank = {d: {v: i for i, v in enumerate(unique_vals[d])} for d in dims}

    def bucketize(d: str, v: Any) -> int:
        n = max(1, len(unique_vals[d]))
        r = rank[d][v]
        # map rank [0..n-1] -> bucket [0..grid_bins_per_dim-1]
        return int(math.floor((r / max(1, n - 1)) * (grid_bins_per_dim - 1))) if n > 1 else 0

    # Group by grid cell
    cells: Dict[tuple, List[int]] = {}
    for o in options:
        key = tuple(bucketize(d, o["meta"].get(d)) for d in dims)
        cells.setdefault(key, []).append(o["option_id"])

    # Sample cells first (spatial locality), then one item from each
    cell_keys = list(cells.keys())
    rng.shuffle(cell_keys)

    selected: List[int] = []
    for key in cell_keys:
        if len(selected) >= k:
            break
        selected.append(rng.choice(cells[key]))

    # If still short, fill randomly from remaining
    if len(selected) < k:
        remaining = [o["option_id"] for o in options if o["option_id"] not in set(selected)]
        rng.shuffle(remaining)
        selected.extend(remaining[: (k - len(selected))])

    return selected[:k]

def csa3_cost_based(options: Sequence[Dict[str, Any]], k: int, rng: random.Random) -> List[int]:
    """
    Cost-based selection: choose lowest capture cost (e.g., shorter shutter).
    Expect options[i]["meta"]["cost"] numeric, smaller=faster.
    Break ties randomly.
    """
    if k >= len(options):
        return [o["option_id"] for o in options]

    # group by cost
    by_cost: Dict[float, List[int]] = {}
    for o in options:
        cost = float(o["meta"].get("cost", 0.0))
        by_cost.setdefault(cost, []).append(o["option_id"])

    costs = sorted(by_cost.keys())
    selected: List[int] = []
    for c in costs:
        ids = by_cost[c]
        rng.shuffle(ids)
        for oid in ids:
            if len(selected) >= k:
                break
            selected.append(oid)
        if len(selected) >= k:
            break
    return selected[:k]
