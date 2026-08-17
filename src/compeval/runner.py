"""Run the eval, compare against a committed baseline, gate the build.

    python -m compeval.runner --provider mock --runs 3
    python -m compeval.runner --provider anthropic --runs 3 --gate
    python -m compeval.runner --provider mock --update-baseline

The gate is the reason this repo exists. An eval that produces a number
nobody blocks a release on is a dashboard, not a test.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import classifiers
from .metrics import Report, score
from .models import Case, CaseOutcome, load_cases
from .report import write_markdown

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASES = ROOT / "data" / "cases.jsonl"
DEFAULT_BASELINE = ROOT / "baselines" / "baseline.json"

# Gate thresholds. Absolute floors first, then no-regression-versus-baseline.
GATE_CRITICAL_RECALL = 1.00   # every must-not-miss case must reach a human
GATE_SURFACING_RECALL = 0.85
GATE_OVER_ESCALATION_RATE = 0.35
GATE_STABILITY = 0.90
GATE_ERROR_RATE = 0.02
REGRESSION_TOLERANCE = 0.03   # allowed drop vs baseline on recall metrics


def run(
    cases: list[Case],
    provider: str,
    runs: int = 1,
    workers: int = 4,
    **kwargs,
) -> tuple[list[CaseOutcome], Report]:
    clf = classifiers.build(provider, **kwargs)
    outcomes = [CaseOutcome(case=c) for c in cases]

    jobs = [(o, _) for o in outcomes for _ in range(runs)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda job: (job[0], clf.classify(job[0].case)), jobs))
    for outcome, prediction in results:
        outcome.predictions.append(prediction)

    return outcomes, score(outcomes, provider=clf.name, runs_per_case=runs)


def evaluate_gate(rep: Report, baseline: dict | None) -> list[str]:
    """Return a list of failure reasons. Empty list means the build passes."""
    failures: list[str] = []

    if rep.critical_recall < GATE_CRITICAL_RECALL:
        failures.append(
            f"critical_recall {rep.critical_recall:.2%} is below the required "
            f"{GATE_CRITICAL_RECALL:.0%}. Auto-resolved: {', '.join(rep.critical_misses) or 'n/a'}"
        )
    if rep.surfacing_recall < GATE_SURFACING_RECALL:
        failures.append(
            f"surfacing_recall {rep.surfacing_recall:.2%} is below floor {GATE_SURFACING_RECALL:.0%}"
        )
    if rep.over_escalation_rate > GATE_OVER_ESCALATION_RATE:
        failures.append(
            f"over_escalation_rate {rep.over_escalation_rate:.2%} exceeds ceiling "
            f"{GATE_OVER_ESCALATION_RATE:.0%} — deflection regression"
        )
    if rep.stability < GATE_STABILITY:
        failures.append(
            f"stability {rep.stability:.2%} is below floor {GATE_STABILITY:.0%}. "
            f"Unstable: {', '.join(rep.unstable_cases[:8]) or 'n/a'}"
        )
    if rep.error_rate > GATE_ERROR_RATE:
        failures.append(f"error_rate {rep.error_rate:.2%} exceeds ceiling {GATE_ERROR_RATE:.0%}")

    if baseline:
        for metric in ("critical_recall", "surfacing_recall", "escalation_recall", "accuracy"):
            was = baseline.get(metric)
            now = getattr(rep, metric)
            if was is None:
                continue
            if now < was - REGRESSION_TOLERANCE:
                failures.append(
                    f"{metric} regressed from {was:.2%} to {now:.2%} "
                    f"(tolerance {REGRESSION_TOLERANCE:.0%})"
                )
        was_oe = baseline.get("over_escalation_rate")
        if was_oe is not None and rep.over_escalation_rate > was_oe + REGRESSION_TOLERANCE:
            failures.append(
                f"over_escalation_rate regressed from {was_oe:.2%} to {rep.over_escalation_rate:.2%}"
            )
    return failures


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="compeval", description=__doc__)
    ap.add_argument(
        "--provider",
        default="mock",
        choices=("mock", "keyword", "anthropic", "ollama", "groq", "openrouter",
                 "together", "openai"),
        help="mock and keyword run offline. ollama runs a real model locally "
             "with no key. groq/openrouter have free tiers.",
    )
    ap.add_argument("--model", default=None,
                    help="override the provider's default model")
    ap.add_argument("--base-url", default=None,
                    help="override the endpoint for OpenAI-compatible providers")
    ap.add_argument("--cases", default=str(DEFAULT_CASES))
    ap.add_argument("--runs", type=int, default=1, help="repeats per case, for stability")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hard-accuracy", type=float, default=0.92,
                    help="mock provider only: accuracy on hard cases. Lower it to\n                          simulate a model regression and watch the gate fire.")
    ap.add_argument("--category", help="filter to one category")
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--baseline", default=str(DEFAULT_BASELINE))
    ap.add_argument("--gate", action="store_true", help="exit non-zero on gate failure")
    ap.add_argument("--update-baseline", action="store_true")
    args = ap.parse_args(argv)

    cases = load_cases(args.cases)
    if args.category:
        cases = [c for c in cases if c.category == args.category]
        if not cases:
            print(f"No cases in category {args.category!r}", file=sys.stderr)
            return 2

    outcomes, rep = run(
        cases,
        provider=args.provider,
        runs=args.runs,
        workers=args.workers,
        seed=args.seed,
        hard_accuracy=args.hard_accuracy,
        **({"model": args.model} if args.model else {}),
        **({"base_url": args.base_url} if args.base_url else {}),
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(rep.to_dict(), indent=2), encoding="utf-8")
    (out_dir / "outcomes.json").write_text(
        json.dumps([o.to_dict() for o in outcomes], indent=2), encoding="utf-8"
    )

    baseline_path = Path(args.baseline)
    baseline = None
    if baseline_path.exists() and not args.update_baseline:
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))

    failures = evaluate_gate(rep, baseline)
    md = write_markdown(rep, outcomes, baseline, failures)
    (out_dir / "report.md").write_text(md, encoding="utf-8")
    print(md)

    if args.update_baseline:
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(json.dumps(rep.to_dict(), indent=2), encoding="utf-8")
        print(f"\nBaseline written to {baseline_path}")
        return 0

    if args.gate and failures:
        print("\nGATE FAILED", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
