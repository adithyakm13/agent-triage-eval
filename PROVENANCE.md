# Provenance

This document exists so that the origin of everything in this repository can be
checked by anyone who asks.

## Summary

Every ticket in `data/cases.jsonl` was written by me for this repository.
Nothing here derives from any company's support transcripts, customer data,
ticketing system, internal documentation, source code, or confidential
knowledge.

## The corpus

The 36 cases are synthetic. Each was written by taking a support-triage failure
mode that is widely discussed in public material — product documentation,
security guidance, published incident write-ups, and general industry
commentary on deploying AI support agents — and composing an original customer
message that exhibits it.

No case is a real customer message, a paraphrase of one, or drawn from any
ticket, transcript, or dataset belonging to any organisation.

The failure modes are general categories, not company-specific:

- Unauthorised financial commitments (refunds, pricing, SLA terms)
- Policy hallucination, including legal and regulatory representations
- Disclosure of another account's data, and credentials pasted into a channel
- Account-takeover and MFA-bypass patterns, including urgency as a social-
  engineering signal
- Legal threats and regulatory complaints
- Enterprise churn signals phrased as routine requests
- Customer hardship and bereavement
- Chargebacks and billing disputes
- Correlated failure reports as an incident signal
- Prompt injection, both direct and disguised as pasted system output

Names, order numbers, seat counts, and account details in the cases are
invented. Any resemblance to a real customer or company is coincidental.

## Labels

Every label is my own judgement, and the reasoning is recorded in the
`rationale` field of each case so that it can be disagreed with. Nobody has
reviewed or approved them. They encode one plausible risk appetite, not an
authoritative standard — see the adaptation notes in the README.

## The code

Written from scratch for this repository. Dependencies are declared in
`pyproject.toml`; the core harness has none beyond the standard library, and
the optional live provider uses the public Anthropic SDK.

## What this repository is not

- Not a description of how any commercial support product works
- Not a benchmark of any commercial product or model
- Not built with, on, or from any employer's infrastructure or data
- Not a production triage system. It is a testing harness.

## Contact

If you believe anything here originates somewhere it shouldn't, raise an issue
or contact me directly and I will address it.
