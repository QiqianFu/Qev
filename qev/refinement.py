# SPDX-License-Identifier: Apache-2.0
# Adapted for Qev; see NOTICE and THIRD_PARTY_NOTICES.md.
"""Teacher-only replay/paired curricula and explicitly parameterized objectives."""
import math
import random



class CyclicSampler:
    def __init__(self, count, seed):
        if count < 1:raise ValueError("empty sampling pool")
        self.count, self.seed, self.cycle, self.position = count, seed, 0, count
        self.order = []

    def take(self):
        if self.position == self.count:
            self.order = list(range(self.count))
            random.Random(self.seed+self.cycle).shuffle(self.order)
            self.cycle += 1;self.position = 0
        value = self.order[self.position];self.position += 1
        return value


def refinement_plan(pair_count, replay_sources, steps, endpoints, replay_fraction, seed):
    if steps < 1 or endpoints < 2 or not 0 <= replay_fraction <= 1:
        raise ValueError("invalid refinement budget")
    replay_n = round(endpoints*replay_fraction)
    if not math.isclose(replay_n,endpoints*replay_fraction) or (endpoints-replay_n)%2:
        raise ValueError("replay fraction must leave a whole number of pairs")
    pairs_n = (endpoints-replay_n)//2
    p = CyclicSampler(pair_count,seed) if pairs_n else None
    r = CyclicSampler(len(replay_sources),seed+100000) if replay_n else None
    plans = []
    for step in range(steps):
        units = [("pair",p.take()) for _ in range(pairs_n)] + [("replay",r.take()) for _ in range(replay_n)]
        random.Random(seed+200000+step).shuffle(units)
        plans.append(units)
    return plans


def microbatches(units, maximum_endpoints):
    if maximum_endpoints < 2:raise ValueError("a microbatch must fit a pair")
    batch, count = [], 0
    for unit in units:
        size = 2 if unit[0] == "pair" else 1
        if batch and count+size > maximum_endpoints:
            yield batch;batch, count = [], 0
        batch.append(unit);count += size
    if batch:yield batch


def distribution_loss(logits, teacher_logits, teacher_temperature=1., student_temperature=1.):
    """Cross entropy against a temperature-adjusted teacher distribution."""
    import torch
    if not all(math.isfinite(v) and v > 0 for v in (teacher_temperature, student_temperature)):
        raise ValueError("temperatures must be finite and positive")
    z = logits.float() / student_temperature
    t = torch.as_tensor(teacher_logits, device=z.device, dtype=torch.float32) / teacher_temperature
    if z.ndim != 1 or z.shape != t.shape or z.numel() < 2:
        raise ValueError("unaligned distribution vectors")
    return -(t.softmax(-1) * z.log_softmax(-1)).sum() * student_temperature**2
