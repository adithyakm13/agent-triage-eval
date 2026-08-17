.PHONY: eval eval-local eval-groq eval-live baseline test compare help

help:
	@echo "make test        unit tests for the scoring layer"
	@echo "make eval        offline deterministic run, gated. No key, no network."
	@echo "make compare     keyword-routing baseline, for context"
	@echo "make eval-local  real model via Ollama. Free, local, no key."
	@echo "make eval-groq   real model via Groq free tier. Needs GROQ_API_KEY."
	@echo "make eval-live   frontier model via Anthropic. Needs ANTHROPIC_API_KEY."
	@echo "make baseline    regenerate baselines/baseline.json from the mock run"

eval:
	PYTHONPATH=src python -m compeval.runner --provider mock --runs 3 --gate

compare:
	PYTHONPATH=src python -m compeval.runner --provider keyword --runs 1 --out results/keyword

# Free, local, no API key. Requires `ollama serve` and `ollama pull llama3.1:8b`.
eval-local:
	PYTHONPATH=src python -m compeval.runner --provider ollama --runs 3 \
		--baseline baselines/baseline-ollama.json --out results/ollama

eval-groq:
	PYTHONPATH=src python -m compeval.runner --provider groq --runs 3 \
		--baseline baselines/baseline-groq.json --out results/groq

eval-live:
	PYTHONPATH=src python -m compeval.runner --provider anthropic --runs 3 --gate \
		--baseline baselines/baseline-live.json --out results/live

baseline:
	PYTHONPATH=src python -m compeval.runner --provider mock --runs 3 --update-baseline

test:
	PYTHONPATH=src python -m pytest -q
