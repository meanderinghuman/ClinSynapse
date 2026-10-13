LIMIT ?= 300
RUN = uv run python -m evals.run --dataset pubmedqa --limit $(LIMIT)

.PHONY: agreement llm-check setup index test smoke baseline evals ablations faults calibration report reproduce serve ui

setup:
	uv sync --group ui --group obs

llm-check:           ## verify the ARC key, list models, test JSON mode
	uv run python -m clinsynapse.llm.check

index:               ## ~40 min on an M-series Mac
	uv run python -m clinsynapse.retrieval.build

test:
	uv run ruff check src evals tests && uv run pytest -q

smoke:               ## record/replay the CI smoke cache (needs an API key on first run)
	uv run python -m evals.run --dataset pubmedqa --limit 30 --workers 1 \
		--cache evals/smoke/cache.sqlite --out reports/ci --name smoke

baseline: smoke      ## promote the current smoke run to the committed CI baseline
	rm -rf reports/baseline/smoke && mkdir -p reports/baseline && cp -r reports/ci/smoke reports/baseline/smoke

ablations:
	uv run python -m evals.run --dataset pubmedqa --retrieval-only
	for p in full single_shot no_rerank no_grader no_verifier bm25 dense alt_model; do \
		$(RUN) --preset $$p; done
	uv run python -m evals.run --dataset router --preset full

faults:
	for f in corrupt_numbers swap_citations remove_gold prompt_injection; do \
		$(RUN) --preset full --fault $$f; done

calibration:          ## optional: needs hand labels (ui/label.py)
	uv run python -m evals.judges.calibration score --judges nli llm-main llm-alt

evals: ablations faults

report:
	uv run python -m evals.report

reproduce: setup index test evals report   ## everything, one command

serve:
	uv run uvicorn clinsynapse.api.app:app --port 8000

ui:
	uv run --group ui streamlit run ui/app.py

agreement:
	uv run python -m evals.judges.agreement reports/runs/pubmedqa-full-n$(LIMIT)
