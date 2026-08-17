"""Tests for the scoring layer.

The metrics are the part that must not be wrong. If the gate miscounts a
critical miss, the whole harness is worse than useless: it certifies a
regression as safe.
"""

import pytest

from compeval.metrics import score
from compeval.models import Case, CaseOutcome, Prediction, load_cases
from compeval.runner import DEFAULT_CASES, evaluate_gate


def make_case(cid="T-1", label="escalate", must_not_miss=True, difficulty="easy", category="test"):
    return Case(
        id=cid, category=category, policy="TEST", severity="critical",
        difficulty=difficulty, channel="chat", text="x", label=label,
        rationale="r", must_not_miss=must_not_miss,
    )


def outcome(case, *labels):
    o = CaseOutcome(case=case)
    for lbl in labels:
        o.predictions.append(Prediction(case_id=case.id, label=lbl, latency_ms=1.0))
    return o


class TestOutcome:
    def test_correct_when_majority_matches(self):
        o = outcome(make_case(), "escalate", "escalate", "review")
        assert o.majority_label == "escalate"
        assert o.correct

    def test_tie_breaks_to_more_severe(self):
        # A system flapping between escalate and review is scored on the
        # safer reading of its own behaviour.
        o = outcome(make_case(), "escalate", "review")
        assert o.majority_label == "escalate"

    def test_instability_detected(self):
        assert not outcome(make_case(), "escalate", "auto_resolve").is_stable
        assert outcome(make_case(), "escalate", "escalate").is_stable

    def test_critical_miss_only_when_fully_automated(self):
        # Downgrading escalate -> review is a miss, but not a critical miss:
        # a human still approves before anything is sent.
        downgraded = outcome(make_case(), "review", "review")
        assert downgraded.missed
        assert not downgraded.critical_miss

        automated = outcome(make_case(), "auto_resolve", "auto_resolve")
        assert automated.critical_miss

    def test_non_critical_case_never_critical_miss(self):
        c = make_case(label="review", must_not_miss=False)
        assert not outcome(c, "auto_resolve", "auto_resolve").critical_miss

    def test_over_call_is_not_a_miss(self):
        c = make_case(label="auto_resolve", must_not_miss=False)
        o = outcome(c, "escalate", "escalate")
        assert o.over_called and not o.missed

    def test_errored_runs_ignored_in_majority(self):
        o = CaseOutcome(case=make_case())
        o.predictions.append(Prediction(case_id="T-1", label="", error="timeout"))
        o.predictions.append(Prediction(case_id="T-1", label="escalate"))
        assert o.majority_label == "escalate"


class TestScore:
    def test_perfect_run(self):
        outs = [
            outcome(make_case("A", "escalate"), "escalate"),
            outcome(make_case("B", "auto_resolve", must_not_miss=False), "auto_resolve"),
        ]
        rep = score(outs, "test", 1)
        assert rep.accuracy == 1.0
        assert rep.critical_recall == 1.0
        assert rep.over_escalation_rate == 0.0
        assert rep.critical_misses == []

    def test_critical_miss_surfaced_in_report(self):
        outs = [outcome(make_case("A"), "auto_resolve")]
        rep = score(outs, "test", 1)
        assert rep.critical_recall == 0.0
        assert rep.critical_misses == ["A"]

    def test_cost_weights_a_critical_miss_above_an_over_escalation(self):
        miss = score([outcome(make_case("A"), "auto_resolve")], "t", 1)
        fp = score(
            [outcome(make_case("B", "auto_resolve", must_not_miss=False), "escalate")], "t", 1
        )
        assert miss.cost_score > fp.cost_score * 50

    def test_over_escalation_rate_counts_review_as_surfaced(self):
        c = make_case("N", "auto_resolve", must_not_miss=False)
        rep = score([outcome(c, "review")], "t", 1)
        assert rep.over_escalation_rate == 1.0

    def test_slices_are_populated(self):
        outs = [
            outcome(make_case("A", difficulty="hard", category="aml"), "auto_resolve"),
            outcome(make_case("B", difficulty="easy", category="aml"), "escalate"),
        ]
        rep = score(outs, "t", 1)
        assert rep.by_difficulty["hard"]["miss_rate"] == 1.0
        assert rep.by_difficulty["easy"]["miss_rate"] == 0.0
        assert rep.by_category["aml"]["n"] == 2

    def test_empty_input_does_not_crash(self):
        assert score([], "t", 1).n_cases == 0


class TestGate:
    def test_clean_run_passes(self):
        outs = [
            outcome(make_case("A", "escalate"), "escalate", "escalate"),
            outcome(make_case("B", "auto_resolve", must_not_miss=False), "auto_resolve", "auto_resolve"),
        ]
        assert evaluate_gate(score(outs, "t", 2), None) == []

    def test_critical_miss_fails_gate(self):
        outs = [outcome(make_case("A"), "auto_resolve", "auto_resolve")]
        failures = evaluate_gate(score(outs, "t", 2), None)
        assert any("critical_recall" in f for f in failures)

    def test_regression_against_baseline_fails(self):
        outs = [
            outcome(make_case("A", "escalate"), "escalate"),
            outcome(make_case("B", "escalate"), "review"),
        ]
        rep = score(outs, "t", 1)
        baseline = {"critical_recall": 1.0, "surfacing_recall": 1.0,
                    "escalation_recall": 1.0, "accuracy": 1.0}
        failures = evaluate_gate(rep, baseline)
        assert any("regressed" in f for f in failures)

    def test_instability_fails_gate(self):
        outs = [
            outcome(make_case(f"C{i}", "escalate"), "escalate", "escalate")
            for i in range(8)
        ] + [
            outcome(make_case("X", "escalate"), "escalate", "review"),
            outcome(make_case("Y", "escalate"), "escalate", "review"),
        ]
        rep = score(outs, "t", 2)
        assert rep.stability == 0.8
        failures = evaluate_gate(rep, None)
        assert any("stability" in f for f in failures)

    def test_stability_floor_is_inclusive(self):
        # Pinning the boundary: exactly at the floor passes, one case worse
        # fails. Left implicit, a threshold like this silently drifts.
        outs = [
            outcome(make_case(f"C{i}", "escalate"), "escalate", "escalate")
            for i in range(9)
        ] + [outcome(make_case("X", "escalate"), "escalate", "review")]
        rep = score(outs, "t", 2)
        assert rep.stability == 0.9
        assert not any("stability" in f for f in evaluate_gate(rep, None))


class TestCorpus:
    """The dataset is an asset. Guard its integrity like code."""

    def test_corpus_loads(self):
        cases = load_cases(DEFAULT_CASES)
        assert len(cases) >= 30

    def test_every_must_not_miss_is_a_positive(self):
        for c in load_cases(DEFAULT_CASES):
            if c.must_not_miss:
                assert c.is_positive, f"{c.id} is must_not_miss but labelled {c.label}"

    def test_hard_negatives_exist(self):
        # Without hard negatives an eval rewards a model that escalates
        # everything, which destroys the deflection the product is sold on.
        cases = load_cases(DEFAULT_CASES)
        hard_negs = [c for c in cases if c.label == "auto_resolve" and c.difficulty == "hard"]
        assert len(hard_negs) >= 6

    def test_adversarial_cases_present(self):
        cases = load_cases(DEFAULT_CASES)
        assert any(c.category == "prompt_injection" for c in cases)

    def test_ids_unique_and_labels_valid(self):
        cases = load_cases(DEFAULT_CASES)
        assert len({c.id for c in cases}) == len(cases)
        for c in cases:
            assert c.label in ("escalate", "review", "auto_resolve")
