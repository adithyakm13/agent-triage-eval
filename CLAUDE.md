# CLAUDE.md

Context for Claude Code working in this repository.

## What this is

An evaluation harness for the triage layer in front of an AI support agent:
given a customer message, decide whether the agent handles it alone, drafts for
human approval, or hands off entirely.

It is a **job-search portfolio artifact** as much as a working tool. It exists
to demonstrate evaluation methodology to founders and hiring managers at
AI-native companies. That dual purpose drives most of the design decisions
below — code quality and the clarity of the README matter more here than
feature count.

## Design decisions that must not be casually reversed

These were deliberate. If you think one is wrong, say so explicitly rather than
quietly changing it.

1. **Three labels, not two.** `auto_resolve` / `review` / `escalate`. Binary
   routing collapses "a human approves before sending" into "a human never sees
   it," which is the distinction the whole repo exists to make.
2. **Asymmetric cost.** A must-not-miss case handled alone costs 100x an
   unnecessary escalation. Accuracy is reported but never gated on.
3. **The gate self-tests.** CI runs a deliberately degraded model and fails the
   build if it *passes* the gate. Don't remove this; it's the part that proves
   the gate works.
4. **Hard negatives adjacent to every positive.** Every escalation case should
   have a near-twin that's correctly automated. Without these the benchmark
   rewards escalating everything. When adding cases, add them in pairs.
5. **Stability is a first-class metric.** Cases run N times; inconsistent
   routing is its own defect class.
6. **Zero required dependencies.** The OpenAI-compatible client uses `urllib`
   from the standard library on purpose, so `pip install -e .` pulls nothing.
   Don't add `requests` or `httpx`.
7. **No API key required.** `make eval` is fully offline. `make eval-local`
   runs a real model through Ollama for free. The owner has no paid API key —
   don't write code or docs that assume one.

## Provenance rules — important

The owner currently works at a company in a regulated-software domain. To keep
this repository clean of any employment conflict:

- **Never** add a case based on a real support interaction, from anywhere.
- **Never** reference the owner's employer, its customers, or its product.
- All cases are synthetic and written from public material. `PROVENANCE.md`
  documents this and must stay accurate as cases are added.
- The repo is **private** until the owner resigns. Don't add anything that
  assumes public visibility (badges pointing at public CI, etc.).

## Layout

```
data/cases.jsonl              36 labelled tickets, one JSON object per line
src/compeval/models.py        Case / Prediction / CaseOutcome, majority + stability
src/compeval/classifiers.py   keyword baseline · offline mock · Anthropic
src/compeval/openai_compat.py Ollama · Groq · OpenRouter · Together · OpenAI
src/compeval/metrics.py       asymmetric cost model, per-slice recall
src/compeval/runner.py        CLI, baseline comparison, gate
src/compeval/report.py        markdown report for PR comments
tests/test_metrics.py         23 tests, including corpus-integrity tests
baselines/baseline.json       committed; changing it is a reviewable diff
```

## Commands

```bash
make test        # 23 unit tests
make eval        # offline, deterministic, gated. No key, no network.
make compare     # keyword-routing baseline
make eval-local  # real model via Ollama, free (ollama serve first)
make baseline    # regenerate baselines/baseline.json
```

Always run `make test && make eval` before committing. The gate should exit 0.

## Known gaps / good next work

- **No frontier-model results.** The owner has no paid API key. The README is
  deliberately explicit about this. If a key becomes available, write results to
  `baselines/baseline-live.json` — never conflate with the offline baseline.
- **Corpus could grow**, but only in matched pairs (see decision 4). 36 cases is
  enough to be credible; volume for its own sake is not the priority.
- **Per-language slicing** exists but there are only 2 non-English cases. More
  would strengthen the multilingual claim.
- **A `results/` directory is gitignored.** Don't commit run outputs.

## Style

Comments explain *why*, not *what*. Several existing comments carry the
reasoning behind a non-obvious choice — those are load-bearing for a reader
evaluating the author's judgement. Preserve that register.
