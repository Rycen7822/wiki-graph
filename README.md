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

Do not reintroduce service restarts, systemd commands, `rag_storage` swaps, old `pending_wikigraph_refresh.json` writes, or direct `custom_kg` live-storage mutation into production paths.

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

Run the production-reference audit:

```bash
python3 -m ops.audit_native_production_refs --repo-root .
```

## Validation gates for changes

For broad changes touching native refresh, wiki integration, query/runtime code, retired-surface audit guards, or test ownership, run the repository test suite from the repo root:

```bash
python3 -m compileall -q tests
python3 -m pytest --collect-only -q tests
PYTHONPATH=. python3 -m pytest tests -q -rs --durations=20
python3 -m compileall -q llm_wiki_native ops tests
git diff --check
```

For smaller owner-family checks, use the current split test files instead of the retired oversized aggregators:

```bash
python3 -m pytest -q \
  tests/test_batch_native_refresh.py \
  tests/test_batch_native_refresh_smoke.py \
  tests/test_batch_native_refresh_preflight.py \
  tests/test_batch_native_refresh_cutover.py \
  tests/test_batch_native_refresh_policy.py

python3 -m pytest -q -rs \
  tests/test_raw_fast_workflows.py \
  tests/test_raw_fast_evidence_bundle.py \
  tests/test_raw_fast_closeout.py

python3 -m pytest -q \
  tests/test_wiki_integration_workflows.py \
  tests/test_wiki_integration_plan.py \
  tests/test_wiki_integration_bridge.py \
  tests/test_batch_wiki_integration_cli.py \
  tests/test_batch_wiki_integration_prompt.py \
  tests/test_batch_wiki_integration_runner.py

python3 -m pytest -q \
  tests/test_wiki_native_lib.py \
  tests/test_wiki_native_production_refs.py \
  tests/test_wiki_native_facade_contract.py \
  tests/test_wiki_native_env_paths.py \
  tests/test_wiki_native_ledger_migration.py

python3 -m pytest -q -rs \
  tests/test_zvec_workspace.py \
  tests/test_zvec_workspace_real.py
```

A bare `python3 -m pytest -q` from the repo root may collect vendored/reference packages that need separate environment setup. Prefer the focused `tests/` gates above unless you are intentionally testing those packages too.

## Safety rules

- Treat the path in `LLM_WIKI_ROOT` as the human wiki, not a scratch directory.
- Use repo-local `wiki_test/` or temporary state directories for destructive tests and rehearsals.
- Keep generated evidence, logs, service output, native state, and sidecars out of the human wiki root.
- Do not clear pending ledgers, cut over workspaces, or promote active pointers without an explicit validation run.
- Prefer native refresh state (`pending_native_refresh.json`) over retired wikigraph refresh state.
- If a command mentions retired wikigraph/custom_kg live-storage mutation, update or remove the caller; production no longer maintains mutation compatibility shims.

## Worknotes

Longer context and historical decisions are kept in `worknotes/`. Review those notes before publishing if your deployment treats operator notes as private.
