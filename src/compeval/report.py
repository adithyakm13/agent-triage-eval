"""Render a run as markdown suitable for a PR comment.

A report a reviewer has to open a JSON file to understand does not get read.
"""

from __future__ import annotations

from .metrics import Report
from .models import CaseOutcome


def write_markdown(
    rep: Report,
    outcomes: list[CaseOutcome],
    baseline: dict | None = None,
    failures: list[str] | None = None,
) -> str:
    failures = failures or []
    lines: list[str] = []

    status = "FAILED" if failures else "PASSED"
    lines.append(f"# Triage eval — {status}")
    lines.append("")
    lines.append(f"Provider `{rep.provider}` · {rep.n_cases} cases · {rep.runs_per_case} run(s) per case")
    lines.append("")

    if failures:
        lines.append("## Gate failures")
        lines.append("")
        for f in failures:
            lines.append(f"- {f}")
        lines.append("")

    lines.append("## Headline")
    lines.append("")
    lines.append("| Metric | Value | Baseline | Note |")
    lines.append("|---|---|---|---|")
    for key, label, note in [
        ("critical_recall", "Critical recall", "must-not-miss cases kept away from the agent — the gate"),
        ("surfacing_recall", "Human-touch recall", "anything a human should see, seen"),
        ("escalation_recall", "Escalation recall", "escalate-labelled cases called correctly"),
        ("over_escalation_rate", "Over-escalation rate", "automatable traffic pushed to a human"),
        ("accuracy", "Exact-label accuracy", "reported, not gated on"),
        ("stability", "Stability", "same input, same answer across runs"),
        ("error_rate", "Error rate", "unusable responses"),
    ]:
        now = getattr(rep, key)
        was = baseline.get(key) if baseline else None
        was_s = f"{was:.1%}" if isinstance(was, (int, float)) else "—"
        lines.append(f"| {label} | {now:.1%} | {was_s} | {note} |")
    lines.append(f"| Weighted cost score | {rep.cost_score:.1f} | "
                 f"{baseline.get('cost_score', '—') if baseline else '—'} | lower is better |")
    lines.append(f"| Mean latency | {rep.mean_latency_ms:.0f} ms | — | p95 {rep.p95_latency_ms:.0f} ms |")
    lines.append("")

    if rep.critical_misses:
        lines.append("## Critical misses")
        lines.append("")
        lines.append("Cases marked must-not-miss that the agent handled alone. "
                     "In production each of these is an output that reached a customer.")
        lines.append("")
        by_id = {o.case.id: o for o in outcomes}
        for cid in rep.critical_misses:
            o = by_id[cid]
            lines.append(f"- **{cid}** ({o.case.category}) — {o.case.rationale}")
        lines.append("")

    if rep.unstable_cases:
        lines.append("## Non-deterministic cases")
        lines.append("")
        lines.append("Same input, different routing across runs. Two customers sending "
                     "the same message get different treatment, unpredictably.")
        lines.append("")
        for cid in rep.unstable_cases:
            got = {p.label for p in next(o for o in outcomes if o.case.id == cid).predictions if p.ok}
            lines.append(f"- **{cid}** — returned {sorted(got)}")
        lines.append("")

    lines.append("## By category")
    lines.append("")
    lines.append("| Category | n | Accuracy | Missed | Over-called |")
    lines.append("|---|---|---|---|---|")
    for name, m in rep.by_category.items():
        lines.append(f"| {name} | {m['n']} | {m['accuracy']:.0%} | {m['missed']} | {m['over_called']} |")
    lines.append("")

    lines.append("## By difficulty")
    lines.append("")
    lines.append("| Difficulty | n | Accuracy | Miss rate |")
    lines.append("|---|---|---|---|")
    for name, m in rep.by_difficulty.items():
        lines.append(f"| {name} | {m['n']} | {m['accuracy']:.0%} | {m['miss_rate']:.0%} |")
    lines.append("")

    return "\n".join(lines)
