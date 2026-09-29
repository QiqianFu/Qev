# SPDX-License-Identifier: Apache-2.0
# Adapted for Qev in 2026; see NOTICE and THIRD_PARTY_NOTICES.md.
"""Deterministic rank allocation without changing optimizer-batch membership."""
import math
import random


ASSIGNMENTS = ("strided-v1", "leaf-balanced-v1")


def late_splice(ids, late, start, seed, epoch, repeats=1):
    """Interleave late record ids uniformly into ids[start:], keeping both orders; ids[:start] is untouched.
    With repeats > 1 the late ids form that many consecutive passes, each in its own shuffled order."""
    head, tail = ids[:start], ids[start:]
    passes = []
    for p in range(repeats):
        order = list(late)
        random.Random(f"late-{seed}-{epoch}" + (f"-{p}" if p else "")).shuffle(order)
        passes += order
    late = passes
    slots = set(random.Random(f"late-slots-{seed}-{epoch}").sample(range(len(tail) + len(late)), len(late)))
    tail, late = iter(tail), iter(late)
    return head + [next(late) if i in slots else next(tail) for i in range(len(ids) - start + len(slots))]


def rank_indices(n, epoch, seed, rank, world, *, batch_size=1, accum=1,
                 assignment="strided-v1", costs=None, late=None, late_repeats=1):
    """late=(start, ids): after shuffling range(n), splice these extra record ids (late_repeats passes of them)
    into global positions >= start.
    With start a multiple of world*batch_size*accum, every earlier optimizer batch is unchanged."""
    if min(n, world, batch_size, accum) < 1 or not 0 <= rank < world:
        raise ValueError("invalid dataset, rank or batch dimensions")
    if assignment not in ASSIGNMENTS:
        raise ValueError(f"unknown batch assignment: {assignment}")
    ids = list(range(n))
    random.Random(seed + epoch).shuffle(ids)
    if late is not None:
        start, extra = late
        if not 0 <= start <= n or set(extra) & set(ids) or len(set(extra)) != len(extra) or late_repeats < 1:
            raise ValueError("late ids must be new, distinct and spliced inside the epoch")
        ids = late_splice(ids, extra, start, seed, epoch, late_repeats)
    n = len(ids)
    size = math.ceil(n / world) * world
    ids += (ids * math.ceil((size - n) / n))[:size - n]
    if assignment == "strided-v1":
        return ids[rank:size:world]
    if costs is None or len(costs) < max(ids) + 1 or any(not math.isfinite(c) or c <= 0 for c in costs):
        raise ValueError("balanced allocation requires one positive finite cost per record")
    result = []
    width = world * batch_size * accum
    for start in range(0, size, width):
        # Balance only inside an existing optimizer step, including padded tails.
        group = list(enumerate(ids[start:start + width]))
        capacity = len(group) // world
        ranks, loads = [[] for _ in range(world)], [0] * world
        for position, index in sorted(group, key=lambda item: (-costs[item[1]], item[0])):
            destination = min((r for r in range(world) if len(ranks[r]) < capacity),
                              key=lambda r: (loads[r], r))
            ranks[destination].append((position, index))
            loads[destination] += costs[index]
        # Preserve the shuffled relative order within each rank.
        result.extend(index for _, index in sorted(ranks[rank]))
    return result
