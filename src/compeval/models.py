"""Core data model for the triage evaluation harness.

Three labels, deliberately not two. A binary handle/escalate model hides the
most important operational decision: the cases an agent may attempt but a
human must approve before anything reaches the customer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterator, Literal

Label = Literal["escalate", "review", "auto_resolve"]
Severity = Literal["critical", "high", "medium", "low"]

LABELS: tuple[Label, ...] = ("escalate", "review", "auto_resolve")

# Ordering matters for "directional" scoring: under-calling is the costly
# direction, over-calling is merely expensive. Handling something alone that
# needed a human is the error that reaches the customer.
LABEL_RANK: dict[str, int] = {"auto_resolve": 0, "review": 1, "escalate": 2}


@dataclass(frozen=True)
class Case:
    """A single labelled support ticket."""

    id: str
    category: str
    policy: str
    severity: Severity
    difficulty: str
    channel: str
    text: str
    label: Label
    rationale: str
    must_not_miss: bool = False

    @property
    def is_positive(self) -> bool:
        """True if a human should be involved at all, in any capacity."""
        return self.label in ("escalate", "review")


@dataclass
class Prediction:
    """What the system under test returned for one case on one run."""

    case_id: str
    label: str
    confidence: float | None = None
    rationale: str = ""
    latency_ms: float = 0.0
    error: str | None = None
    raw: str = ""

    @property
    def ok(self) -> bool:
        return self.error is None and self.label in LABELS


@dataclass
class CaseOutcome:
    """A case joined to its prediction(s), with the verdict already computed."""

    case: Case
    predictions: list[Prediction] = field(default_factory=list)

    @property
    def majority_label(self) -> str:
        """Modal prediction across runs. Ties resolve to the most severe label,
        because a system that is unstable between 'escalate' and 'review'
        should be scored on the safer reading of its own behaviour."""
        counts: dict[str, int] = {}
        for p in self.predictions:
            if p.ok:
                counts[p.label] = counts.get(p.label, 0) + 1
        if not counts:
            return "error"
        top = max(counts.values())
        tied = [lbl for lbl, c in counts.items() if c == top]
        return max(tied, key=lambda l: LABEL_RANK.get(l, -1))

    @property
    def is_stable(self) -> bool:
        """Did every successful run agree? Non-determinism is a defect class of
        its own: two customers sending the same message should not be routed
        differently depending on which way the sampling fell."""
        labels = {p.label for p in self.predictions if p.ok}
        return len(labels) <= 1

    @property
    def correct(self) -> bool:
        return self.majority_label == self.case.label

    @property
    def missed(self) -> bool:
        """Under-called: the system rated this less serious than it is."""
        got = LABEL_RANK.get(self.majority_label, -1)
        want = LABEL_RANK.get(self.case.label, -1)
        return got < want

    @property
    def over_called(self) -> bool:
        got = LABEL_RANK.get(self.majority_label, -1)
        want = LABEL_RANK.get(self.case.label, -1)
        return got > want

    @property
    def critical_miss(self) -> bool:
        """The failure that reaches the customer: a must-not-miss case the
        agent handled entirely on its own."""
        return self.case.must_not_miss and self.majority_label == "auto_resolve"

    @property
    def mean_latency_ms(self) -> float:
        vals = [p.latency_ms for p in self.predictions if p.ok]
        return sum(vals) / len(vals) if vals else 0.0

    def to_dict(self) -> dict:
        return {
            "case_id": self.case.id,
            "category": self.case.category,
            "severity": self.case.severity,
            "difficulty": self.case.difficulty,
            "expected": self.case.label,
            "got": self.majority_label,
            "correct": self.correct,
            "missed": self.missed,
            "over_called": self.over_called,
            "critical_miss": self.critical_miss,
            "stable": self.is_stable,
            "runs": [asdict(p) for p in self.predictions],
        }


def load_cases(path: str | Path) -> list[Case]:
    """Read the JSONL corpus, failing loudly on a malformed line."""
    cases: list[Case] = []
    seen: set[str] = set()
    for lineno, raw in enumerate(_iter_lines(Path(path)), start=1):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{lineno} is not valid JSON: {exc}") from exc
        try:
            case = Case(**payload)
        except TypeError as exc:
            raise ValueError(f"{path}:{lineno} does not match the Case schema: {exc}") from exc
        if case.label not in LABELS:
            raise ValueError(f"{path}:{lineno} has unknown label {case.label!r}")
        if case.id in seen:
            raise ValueError(f"{path}:{lineno} duplicates case id {case.id!r}")
        seen.add(case.id)
        cases.append(case)
    if not cases:
        raise ValueError(f"No cases found in {path}")
    return cases


def _iter_lines(path: Path) -> Iterator[str]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("//"):
                yield line
