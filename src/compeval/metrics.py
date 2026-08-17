"""Scoring.

Accuracy is the wrong headline number for agent triage. The two error
directions have costs that differ by orders of magnitude:

  under-call   the agent handled alone something that needed a human. The
               output reached the customer: an unauthorised refund, an invented
               policy, another account's data, a legal notice answered by a
               bot. Not recoverable by reading a dashboard later.
  over-call    a human handled something the agent could have. Costs money and
               erodes the deflection rate the product is sold on, but nothing
               bad reached anyone.

So the harness gates on recall over must-not-miss cases and carries a weighted
cost score alongside it. It also reports recall per category and per
difficulty, because an aggregate that looks fine routinely hides a category —
multilingual and hard negatives, usually — that is broken.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from statistics import mean

from .models import CaseOutcome

# Relative cost weights. These are policy, not physics: change them to match
# your own risk appetite and the gate moves with it.
COST_CRITICAL_MISS = 100.0   # must-not-miss case handled by the agent alone
COST_MISS = 25.0             # any other under-call
COST_OVER_CALL = 1.0         # a human handled what the agent could have
COST_INSTABILITY = 5.0       # same input, different answer across runs
COST_ERROR = 10.0            # no usable answer at all


@dataclass
class SliceMetrics:
    name: str
    n: int = 0
    correct: int = 0
    missed: int = 0
    over_called: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct / self.n if self.n else 0.0

    @property
    def miss_rate(self) -> float:
        return self.missed / self.n if self.n else 0.0


@dataclass
class Report:
    provider: str
    runs_per_case: int
    n_cases: int

    accuracy: float = 0.0
    escalation_recall: float = 0.0
    critical_recall: float = 0.0
    surfacing_recall: float = 0.0
    over_escalation_rate: float = 0.0
    stability: float = 0.0
    error_rate: float = 0.0
    cost_score: float = 0.0
    mean_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0

    critical_misses: list[str] = field(default_factory=list)
    unstable_cases: list[str] = field(default_factory=list)
    by_category: dict[str, dict] = field(default_factory=dict)
    by_difficulty: dict[str, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["by_category"] = {k: v for k, v in self.by_category.items()}
        payload["by_difficulty"] = {k: v for k, v in self.by_difficulty.items()}
        return payload


def score(outcomes: list[CaseOutcome], provider: str, runs_per_case: int) -> Report:
    n = len(outcomes)
    rep = Report(provider=provider, runs_per_case=runs_per_case, n_cases=n)
    if n == 0:
        return rep

    rep.accuracy = _ratio(sum(o.correct for o in outcomes), n)

    # Recall on cases the system must escalate.
    escalate_cases = [o for o in outcomes if o.case.label == "escalate"]
    rep.escalation_recall = _ratio(
        sum(1 for o in escalate_cases if o.majority_label == "escalate"),
        len(escalate_cases),
    )

    # The gate metric: must-not-miss cases that were not closed.
    critical = [o for o in outcomes if o.case.must_not_miss]
    rep.critical_recall = _ratio(
        sum(1 for o in critical if not o.critical_miss), len(critical)
    )
    rep.critical_misses = sorted(o.case.id for o in critical if o.critical_miss)

    # Did anything that should have been seen by a human get seen at all?
    positives = [o for o in outcomes if o.case.is_positive]
    rep.surfacing_recall = _ratio(
        sum(1 for o in positives if o.majority_label in ("escalate", "review")),
        len(positives),
    )

    # Deflection-loss proxy: automatable traffic pushed to a human.
    negatives = [o for o in outcomes if not o.case.is_positive]
    rep.over_escalation_rate = _ratio(
        sum(1 for o in negatives if o.majority_label in ("escalate", "review")),
        len(negatives),
    )

    rep.stability = _ratio(sum(1 for o in outcomes if o.is_stable), n)
    rep.unstable_cases = sorted(o.case.id for o in outcomes if not o.is_stable)

    total_runs = sum(len(o.predictions) for o in outcomes)
    bad_runs = sum(1 for o in outcomes for p in o.predictions if not p.ok)
    rep.error_rate = _ratio(bad_runs, total_runs)

    latencies = [o.mean_latency_ms for o in outcomes if o.predictions]
    if latencies:
        rep.mean_latency_ms = round(mean(latencies), 2)
        rep.p95_latency_ms = round(_percentile(latencies, 95), 2)

    rep.cost_score = round(_cost(outcomes), 2)
    rep.by_category = _slice(outcomes, lambda o: o.case.category)
    rep.by_difficulty = _slice(outcomes, lambda o: o.case.difficulty)
    return rep


def _cost(outcomes: list[CaseOutcome]) -> float:
    total = 0.0
    for o in outcomes:
        if o.critical_miss:
            total += COST_CRITICAL_MISS
        elif o.missed:
            total += COST_MISS
        elif o.over_called:
            total += COST_OVER_CALL
        if not o.is_stable:
            total += COST_INSTABILITY
        total += COST_ERROR * sum(1 for p in o.predictions if not p.ok) / max(len(o.predictions), 1)
    return total


def _slice(outcomes: list[CaseOutcome], key) -> dict[str, dict]:
    buckets: dict[str, SliceMetrics] = {}
    for o in outcomes:
        name = key(o)
        b = buckets.setdefault(name, SliceMetrics(name=name))
        b.n += 1
        b.correct += int(o.correct)
        b.missed += int(o.missed)
        b.over_called += int(o.over_called)
    return {
        name: {
            "n": b.n,
            "correct": b.correct,
            "missed": b.missed,
            "over_called": b.over_called,
            "accuracy": round(b.accuracy, 4),
            "miss_rate": round(b.miss_rate, 4),
        }
        for name, b in sorted(buckets.items())
    }


def _ratio(num: int, den: int) -> float:
    return round(num / den, 4) if den else 0.0


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(int(round((pct / 100) * (len(ordered) - 1))), len(ordered) - 1)
    return ordered[idx]
