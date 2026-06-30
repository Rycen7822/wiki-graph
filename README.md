# Wiki Graph

This repository contains the native zvec runtime, refresh tooling, query helpers, tests, and rehearsal workspace for an `llm-wiki` graph/retrieval pipeline.

Production retrieval is native-only. The previous `wikigraph` / `custom_kg` live-storage backend has been retired from production paths; remaining old-named scripts are explicit read-only diagnostics or fail-closed shims so stale automation cannot mutate live storage by accident.

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

- `scripts/batch_native_refresh.py` manages pending native refresh state, workspace preparation, and guarded native cutover.
- `scripts/batch_wiki_integration.py` manages raw-fast notes waiting to be integrated into the human wiki before graph refresh.
- `scripts/wiki_search.py` queries the native service/runtime and can save evidence packs.
- `llm-wiki-native/` contains the native package, API server, zvec/sqlite storage code, retrieval engine, manifest handling, pointer handling, and native tests.

Retired compatibility surface:

- `scripts/batch_wikigraph_refresh.py` is only a retired wrapper. Status/decision commands are read-only; mutation commands fail closed and point to native refresh state.
- `scripts/custom_kg_incremental.py` still has manifest/export helper commands used by native staging, but old live-storage commands (`plan`, `audit-storage`, `apply`, `materialize-full`, `finalize-prepared-swap`) are retired/fail-closed.
- `scripts/import_custom_kg.py` is a retired cold-import shim. `--dry-run` can summarize payload shape; non-dry import and direct old graph construction fail closed.
- `scripts/wiki_wikigraph_refresh_pending.py` can read old pending ledger state for diagnostics, but writing old wikigraph refresh ledgers is retired/fail-closed.

Do not reintroduce service restarts, systemd commands, `rag_storage` swaps, old `pending_wikigraph_refresh.json` writes, or direct `custom_kg` live-storage mutation into production paths.

## Common commands

Check native refresh state:

```bash
python3 scripts/batch_native_refresh.py status \
  --root "$LLM_WIKI_ROOT" \
  --state-dir "$LLM_WIKI_STATE_DIR" \
  --workdir "$WIKI_GRAPH_REPO"
```

Check wiki-integration state:

```bash
python3 scripts/batch_wiki_integration.py status \
  --root "$LLM_WIKI_ROOT" \
  --state-dir "$LLM_WIKI_STATE_DIR" \
  --reason threshold
```

Mark native refresh as pending when upstream integration or reviewed changes require a graph refresh:

```bash
python3 scripts/batch_native_refresh.py mark-pending \
  --root "$LLM_WIKI_ROOT" \
  --state-dir "$LLM_WIKI_STATE_DIR" \
  --workdir "$WIKI_GRAPH_REPO" \
  --reason manual
```

Prepare a native workspace without cutover:

```bash
python3 scripts/batch_native_refresh.py refresh \
  --root "$LLM_WIKI_ROOT" \
  --state-dir "$LLM_WIKI_STATE_DIR" \
  --workdir "$WIKI_GRAPH_REPO" \
  --prepare-only
```

Cutover is intentionally explicit and should only be done after validation gates pass:

```bash
python3 scripts/batch_native_refresh.py refresh \
  --root "$LLM_WIKI_ROOT" \
  --state-dir "$LLM_WIKI_STATE_DIR" \
  --workdir "$WIKI_GRAPH_REPO" \
  --cutover
```

Query the native service/data endpoint:

```bash
python3 scripts/wiki_search.py "your query" \
  --root "$LLM_WIKI_ROOT" \
  --state-dir "$LLM_WIKI_STATE_DIR" \
  --data-only
```

Run the production-reference audit:

```bash
python3 scripts/audit_native_production_refs.py --repo-root .
```

## Validation gates for changes

For changes touching native refresh, wiki integration, query/runtime code, or retired compatibility shims, run the focused repo tests:

```bash
python3 -m pytest \
  tests/test_batch_native_refresh.py \
  tests/test_wiki_native_lib.py \
  tests/test_wikigraph_refresh.py \
  tests/test_wiki_wikigraph_compat_lib.py \
  tests/test_custom_kg_materialize.py \
  tests/test_vector_cache.py \
  tests/test_native_zvec_materialize.py \
  -q
```

Run native package tests with its source directory on `PYTHONPATH`:

```bash
PYTHONPATH=llm-wiki-native/src python3 -m pytest llm-wiki-native/tests -q
```

Run syntax and whitespace checks before committing:

```bash
python3 -m py_compile \
  scripts/audit_native_production_refs.py \
  scripts/batch_native_refresh.py \
  scripts/batch_wikigraph_refresh.py \
  scripts/import_custom_kg.py \
  scripts/custom_kg_incremental.py \
  scripts/custom_kg_materialize.py \
  scripts/custom_kg_vector_fill.py \
  scripts/native_zvec_materialize.py \
  scripts/wiki_wikigraph_refresh_pending.py \
  scripts/wiki_wikigraph_compat_lib.py \
  scripts/wiki_native_cli.py \
  scripts/wiki_native_wiki_integration_pending.py \
  scripts/raw_fast_closeout.py \
  scripts/raw_fast_evidence_bundle.py \
  scripts/vector_cache.py

git diff --check
```

A bare `python3 -m pytest -q` from the repo root may collect vendored/reference packages that need separate environment setup. Prefer the focused gates above unless you are intentionally testing those packages too.

## Safety rules

- Treat the path in `LLM_WIKI_ROOT` as the human wiki, not a scratch directory.
- Use repo-local `wiki_test/` or temporary state directories for destructive tests and rehearsals.
- Keep generated evidence, logs, service output, native state, and sidecars out of the human wiki root.
- Do not clear pending ledgers, cut over workspaces, or promote active pointers without an explicit validation run.
- Prefer native refresh state (`pending_native_refresh.json`) over retired wikigraph refresh state.
- If a command mentions retired wikigraph/custom_kg live-storage mutation, expect it to fail closed; update the caller to use native refresh instead.

## Worknotes

Longer context and historical decisions are kept in `worknotes/`. Review those notes before publishing if your deployment treats operator notes as private.
