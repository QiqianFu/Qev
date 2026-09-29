# SPDX-License-Identifier: Apache-2.0
# Adapted for Qev in 2026; see NOTICE and THIRD_PARTY_NOTICES.md.
from collections import Counter

import pytest

from qev.batching import rank_indices


@pytest.mark.parametrize("n,world,batch,accum", [(97, 2, 4, 2), (23, 4, 2, 2), (1, 4, 1, 1)])
def test_balancing_preserves_every_optimizer_batch_and_padding(n, world, batch, accum):
    costs = [(i % 7 + 1) ** 3 for i in range(n)]
    args = dict(batch_size=batch, accum=accum)
    original = [rank_indices(n, 2, 17, r, world, **args) for r in range(world)]
    balanced = [rank_indices(n, 2, 17, r, world, assignment="leaf-balanced-v1", costs=costs, **args)
                for r in range(world)]
    assert len({len(ids) for ids in balanced}) == 1
    stride = batch * accum
    for start in range(0, len(original[0]), stride):
        assert Counter(i for rank in original for i in rank[start:start + stride]) == Counter(
            i for rank in balanced for i in rank[start:start + stride])
    assert balanced[0] == rank_indices(n, 2, 17, 0, world, assignment="leaf-balanced-v1", costs=costs, **args)


def test_cost_allocation_reduces_work_on_heavier_rank():
    n = 32
    order = rank_indices(n, 0, 17, 0, 1)
    costs = [1] * n
    for i in order[::2]:
        costs[i] = 100
    original = [rank_indices(n, 0, 17, r, 2) for r in range(2)]
    balanced = [rank_indices(n, 0, 17, r, 2, batch_size=16, assignment="leaf-balanced-v1", costs=costs)
                for r in range(2)]
    assert max(sum(costs[i] for i in r) for r in balanced) < max(sum(costs[i] for i in r) for r in original)
    assert len(balanced[0]) == len(balanced[1]) == 16


def test_reject_bad_allocation_configuration():
    with pytest.raises(ValueError, match="unknown batch"):
        rank_indices(3, 0, 17, 0, 2, assignment="typo")
    with pytest.raises(ValueError, match="positive finite"):
        rank_indices(3, 0, 17, 0, 2, assignment="leaf-balanced-v1", costs=[1, 2, float("nan")])


@pytest.mark.parametrize("assignment", ["strided-v1", "leaf-balanced-v1"])
def test_late_splice_keeps_earlier_optimizer_batches(assignment):
    n, world, batch, accum = 101, 4, 3, 2
    width = world * batch * accum
    extra = list(range(n, n + 17))
    costs = [(i % 5 + 1) ** 2 for i in range(n + len(extra))]
    args = dict(batch_size=batch, accum=accum, assignment=assignment, costs=costs)
    start = 2 * width
    plain = [rank_indices(n, 1, 17, r, world, **args) for r in range(world)]
    mixed = [rank_indices(n, 1, 17, r, world, late=(start, extra), **args) for r in range(world)]
    assert all(m[:start // world] == p[:start // world] for m, p in zip(mixed, plain))
    tail = Counter(i for m in mixed for i in m[start // world:])
    assert all(tail[i] == 1 for i in extra) and not Counter(i for m in mixed for i in m[:start // world]) & Counter(extra)
    assert len({len(m) for m in mixed}) == 1 and sum(map(len, mixed)) == 120  # 118 padded to a multiple of 4
    assert mixed == [rank_indices(n, 1, 17, r, world, late=(start, extra), **args) for r in range(world)]
    with pytest.raises(ValueError, match="late ids"):
        rank_indices(n, 1, 17, 0, world, late=(start, [0]), **args)


def test_late_repeats_form_shuffled_passes_and_keep_single_pass_order():
    n, world, batch, accum = 101, 4, 3, 2
    extra = list(range(n, n + 17))
    start = 2 * world * batch * accum
    one = [rank_indices(n, 1, 17, r, world, batch_size=batch, accum=accum, late=(start, extra)) for r in range(world)]
    three = [rank_indices(n, 1, 17, r, world, batch_size=batch, accum=accum, late=(start, extra), late_repeats=3)
             for r in range(world)]
    assert all(t[:start // world] == o[:start // world] for t, o in zip(three, one))
    counts = Counter(i for t in three for i in t)
    assert all(counts[i] == 3 for i in extra) and sum(map(len, three)) == 152  # 101 + 51
    merged = [i for pos in range(len(three[0])) for t in three for i in t[pos:pos + 1]]
    order = [i for i in merged if i >= n]
    assert all(sorted(order[k * 17:(k + 1) * 17]) == extra for k in range(3))  # consecutive complete passes
