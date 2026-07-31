# Wiki Graph

This repository contains the native zvec runtime, refresh tooling, query helpers, tests, and rehearsal workspace for an `llm-wiki` graph/retrieval pipeline.

Production retrieval is native-only. The previous `wikigraph` / `custom_kg` live-storage backend has been retired from production paths; stale live-mutation automation should fail through missing entrypoints or explicit audit detectors, not maintained compatibility shims.

## Configure your environment

This README intentionally uses environment variables instead of machine-specific paths. Set these values for your deployment before running the examples:

```bash
export LLM_WIKI_ROOT="/path/to/your/wiki"
export LLM_WIKI_STATE_DIR="/path/to/your/native-zvec-state"
export WIKI_GRAPH_REPO="$(pwd)"
```

Recommended conventions:

- `LLM_WIKI_ROOT` points at the human-edited wiki content.
- `LLM_WIKI_STATE_DIR` points at generated native zvec state and pending refresh ledgers.
- `WIKI_GRAPH_REPO` points at this repository checkout.
- Use repo-local `wiki_test/` or a throwaway temp directory for rehearsals and destructive tests.
- Keep generated state, logs, evidence packs, temporary files, and sidecar artifacts outside the human wiki root unless a workflow explicitly says it writes wiki content.

## What is active vs retired

Active production owner:

- `ops.batch_native_refresh` manages pending native refresh state, workspace preparation, and guarded native cutover.
- `ops.batch_wiki_integration` manages raw-fast notes waiting to be integrated into the human wiki before graph refresh.
- `ops.wiki_search` queries the native service/runtime and can save evidence packs.
- `llm_wiki_native/` contains the native package, API server, zvec/sqlite storage code, retrieval engine, manifest handling, and pointer handling.

Retired live-storage boundary:

- `ops.custom_kg_incremental` exposes only native manifest helper commands (`export-manifest`, `audit-manifest-content`); old live-storage commands were removed after native cutover.

Do not reintroduce service restarts, systemd commands, `rag_storage` swaps, retired wikigraph refresh ledger writes, or direct `custom_kg` live-storage mutation into production paths.

## Common commands

Check native refresh state:

```bash
python3 -m ops.batch_native_refresh status \
  --root "$LLM_WIKI_ROOT" \
  --state-dir "$LLM_WIKI_STATE_DIR" \
  --workdir "$WIKI_GRAPH_REPO"
```

Check wiki-integration state:

```bash
python3 -m ops.batch_wiki_integration status \
  --root "$LLM_WIKI_ROOT" \
  --state-dir "$LLM_WIKI_STATE_DIR" \
  --reason threshold
```

Mark native refresh as pending when upstream integration or reviewed changes require a graph refresh:

```bash
python3 -m ops.batch_native_refresh mark-pending \
  --root "$LLM_WIKI_ROOT" \
  --state-dir "$LLM_WIKI_STATE_DIR" \
  --workdir "$WIKI_GRAPH_REPO" \
  --reason manual
```

Prepare a native workspace without cutover:

```bash
python3 -m ops.batch_native_refresh refresh \
  --root "$LLM_WIKI_ROOT" \
  --state-dir "$LLM_WIKI_STATE_DIR" \
  --workdir "$WIKI_GRAPH_REPO" \
  --prepare-only
```

Cutover is intentionally explicit and should only be done after validation gates pass:

```bash
python3 -m ops.batch_native_refresh refresh \
  --root "$LLM_WIKI_ROOT" \
  --state-dir "$LLM_WIKI_STATE_DIR" \
  --workdir "$WIKI_GRAPH_REPO" \
  --cutover
```

Query the native service/data endpoint:

```bash
python3 -m ops.wiki_search "your query" \
  --root "$LLM_WIKI_ROOT" \
  --state-dir "$LLM_WIKI_STATE_DIR" \
  --data-only
```

### Relevance-aware retrieval

Set `NATIVE_SERVER` to the native API endpoint. Focused retrieval favors the
smallest decisive evidence set:

```bash
python3 -m ops.wiki_search \
  "Explain the decisive result and its supporting measurements." \
  --server "$NATIVE_SERVER" \
  --retrieval-goal focused \
  --top-k 4 \
  --response-profile standard \
  --data-only \
  --no-record-query-event
```

Coverage retrieval preserves distinct evidence across sources:

```bash
python3 -m ops.wiki_search \
  "Compare the two approaches, including evidence and limitations from each." \
  --server "$NATIVE_SERVER" \
  --retrieval-goal coverage \
  --top-k 6 \
  --response-profile debug \
  --data-only \
  --no-record-query-event
```

`top_k` is the maximum number of visible context blocks, not the number of
records initially retrieved. The planner uses a larger internal candidate
budget before selection; that budget is an implementation detail, not a public
tuning knob.

Responses using `ranking_contract="relevance-v1"` expose `score` as the
weighted relevance score used for final ordering. It combines local route rank,
source rank, normalized query-term coverage, and evidence quality; it is not
the legacy raw route score. Use `relevance_score_breakdown` and `route_ranks`
to interpret it. Existing debug hits, coverage gaps, and `score_breakdown`
remain as additive compatibility fields, but `score_breakdown` must not be used
to reconstruct the relevance-v1 score.

To collect a candidate quality report, point the variables below at an audited
workspace, a frozen relevance-v1 suite, its matching baseline, and a new output
path:

```bash
python3 -m ops.collect_native_query_report \
  --quality-contract relevance-v1 \
  --partition all \
  --query-suite "$RELEVANCE_QUERY_SUITE" \
  --workspace-file "$NATIVE_WORKSPACE_FILE" \
  --runtime-code-root "$WIKI_GRAPH_REPO" \
  --baseline-report "$RELEVANCE_BASELINE_REPORT" \
  --server "$NATIVE_SERVER" \
  --endpoint /query/data \
  --warmup-runs 1 \
  --repetitions 5 \
  --require-gates \
  --fail-if-output-exists \
  --output "$RELEVANCE_CANDIDATE_REPORT"
```

This retrieval-only upgrade uses the existing audited workspace and does not
require a workspace rebuild, native refresh, or pointer cutover. Those
operations remain separate workflows for corpus or production-state changes.

Run the production-reference audit:

```bash
python3 -m ops.audit_native_production_refs --repo-root .
```

## Validation gates for changes

For broad changes touching native refresh, wiki integration, query/runtime code, retired-surface audit guards, or test ownership, run the repository test suite from the repo root:

```bash
python3 -m compileall -q tests
PYTHONPATH=. python3 -m pytest --collect-only -q tests
PYTHONPATH=. python3 -m pytest tests -q -rs --durations=20
python3 -m compileall -q llm_wiki_native ops tests
git diff --check
```

For smaller owner-family checks, use the current split test files instead of the retired oversized aggregators:

```bash
PYTHONPATH=. python3 -m pytest -q \
  tests/test_batch_native_refresh.py \
  tests/test_batch_native_refresh_smoke.py \
  tests/test_batch_native_refresh_preflight.py \
  tests/test_batch_native_refresh_cutover.py \
  tests/test_batch_native_refresh_policy.py

PYTHONPATH=. python3 -m pytest -q -rs \
  tests/test_raw_fast_workflows.py \
  tests/test_raw_fast_evidence_bundle.py \
  tests/test_raw_fast_evidence_resources.py \
  tests/test_raw_fast_evidence_sources.py \
  tests/test_raw_fast_evidence_figures.py \
  tests/test_raw_fast_closeout.py

PYTHONPATH=. python3 -m pytest -q \
  tests/test_wiki_integration_workflows.py \
  tests/test_batch_wiki_integration_prompt.py \
  tests/test_batch_wiki_integration_runner.py

PYTHONPATH=. python3 -m pytest -q \
  tests/test_package_boundaries.py \
  tests/test_wiki_native_production_refs.py \
  tests/test_wiki_native_facade_contract.py \
  tests/test_wiki_native_env_paths.py

PYTHONPATH=. python3 -m pytest -q -rs \
  tests/test_zvec_workspace.py \
  tests/test_zvec_workspace_real.py
```
