# LightRAG compact refresh log fix design audit

## 1. Request normalization

User asked to fix oversized LightRAG refresh logs so future refresh logs keep only key information. The observed incident is a `refresh_logs/*_batch_lightrag_refresh_import.log` file around 1.27GB after `full_materialization + reuse_vector_cache`.

Desired outcome:

- Full-materialization refresh still uses vector cache and produces identical storage/manifest behavior.
- Default stdout, prepared-swap report, and batch import log no longer include per-record embedding vectors.
- Compact report keeps action-critical anchors: import mode, payload/manifest counts, cache path, blocker summary, fill flag, seed/fill summaries, vector hit/miss summaries, audit results, paths, timings, hashes, and finalize command.
- No external fallback, no service behavior change, no storage format change.

## 2. Sources inspected

- `/home/xu/project/wiki/lightrag/state/refresh_logs/20260627_222032_batch_lightrag_refresh_import.log`: 1,266,848,950 bytes; contained 38,041 occurrences of `"vector": [`.
- `/home/xu/project/wiki/lightrag/state/refresh_logs/20260627_161035_batch_lightrag_refresh_import.log`: 1,253,649,093 bytes; contained 37,636 occurrences of `"vector": [`.
- `/home/xu/project/wiki/lightrag/scripts/vector_cache.py:285-349`: `resolve_manifest_vectors()` returns `resolved` entries with `vector` arrays plus `missing` and `summary`.
- `/home/xu/project/wiki/wiki-graph/scripts/custom_kg_incremental.py:1766-1810`: `run_full_materialization_no_swap()` includes the full `vector_report` in `report`, then prints/writes that report.
- `/home/xu/project/wiki/wiki-graph/scripts/custom_kg_incremental.py:318-329`: `write_prepared_swap_bundle()` writes the report to `state/prepared_swaps/custom_kg_prepared_swap.json`.
- `/home/xu/project/wiki/wiki-graph/scripts/batch_lightrag_refresh.py:271-274`: `run_subprocess()` appends subprocess stdout directly to import log.
- `/home/xu/project/wiki/wiki-graph/tests/test_custom_kg_materialize.py:305-466`: existing materialize-full tests check summary/audit behavior and can be extended.

## 3. Design iteration logs

### Iteration 1 — root cause

1. What makes the log large? `materialize-full` stdout contains the full `vector_report["resolved"]` tree, including 38,041 embedding arrays.
2. Is the final current import report also huge? No. `custom_kg_import_report.json` after finalize is around 3KB because finalize writes a compact final report.

Research added: live log size/count probes and source reads.
Design change: target the materialize-full report construction seam rather than refresh scheduling.
Remaining uncertainty: none for root cause.

### Iteration 2 — minimal owner seam

1. Does materialization need the full vector report? Yes, `materialize_file_storage_from_manifest()` needs `vector_report["resolved"]` in memory before report emission.
2. Does default report/log need resolved vectors? No; action-critical validation only needs summary, miss counts/examples, seed/fill summary, and audits.

Research added: `resolve_manifest_vectors()` contract and existing tests.
Design change: keep full vectors in memory; replace report-facing `vector_cache` with a compact projection.
Remaining uncertainty: if a future debug mode needs full vectors, add an explicit debug artifact later; do not keep full vectors by default.

## 4. Baseline design inventory

| id | existing element | current assumption/contract | evidence | owner/seam | risk if changed |
|---|---|---|---|---|---|
| B1 | `resolve_manifest_vectors()` | Returns full resolved vectors for materialization plus summary/missing diagnostics. | `vector_cache.py:285-349` | vector-cache data seam | Removing vectors here would break shadow materialization. |
| B2 | `run_full_materialization_no_swap()` report | Report currently includes full `vector_report`, so stdout/prepared report/log become huge. | `custom_kg_incremental.py:1766-1810`; live log counts | materialize-full reporting seam | Keeping this creates GB logs for every full+reuse run. |
| B3 | `batch_lightrag_refresh.py run_subprocess()` | Captures subprocess stdout verbatim into refresh logs. | `batch_lightrag_refresh.py:271-274` | batch log seam | Broad truncation could hide real failures; root cause should be fixed upstream. |
| B4 | Prepared-swap bundle | Must carry finalize-critical paths/hashes/audits, not raw vectors. | M1 prepared-swap reference; current report fields | prepared swap report | Removing needed paths/hashes would break finalize. |
| B5 | Existing tests | Tests mostly assert summaries/audits, not resolved vectors. | `tests/test_custom_kg_materialize.py` | regression tests | A false green could miss prepared report bloat. |

## 5. Proposed design ledger

| id | baseline refs | proposed decision | intent | files/seams touched | expected impact | rollback/proof |
|---|---|---|---|---|---|---|
| D1 | B1, B2 | Add a compact vector report projection in `custom_kg_incremental.py`; keep full `vector_report` only in memory. | Eliminate vector arrays from stdout/prepared reports while preserving materialization. | `scripts/custom_kg_incremental.py` | Full+reuse logs shrink from GB to KB/MB scale. | Tests assert no `resolved`/`vector` keys in report JSON and materialized shadow still audits ok. |
| D2 | B4 | Prepared-swap reports use the same compact vector report. | Prevent `state/prepared_swaps/custom_kg_prepared_swap.json` from becoming GB-sized. | `scripts/custom_kg_incremental.py` | Prepared report remains finalize-safe and compact. | Test reads prepared report JSON and asserts compact fields and missing raw vectors. |
| D3 | B5 | Extend existing materialization tests rather than adding a new large fixture suite. | Keep proof small and domain-neutral. | `tests/test_custom_kg_materialize.py` | Protects exact regression with small vectors. | Focused pytest gate. |
| D4 | B3 | Do not add broad subprocess stdout truncation in this slice. | Avoid hiding failures and fix the owning report seam first. | none | Batch logs become compact because subprocess output is compact. | If future commands still over-log, add command-specific log compaction separately. |

## 6. Compression review

| id | baseline refs | decision refs | compression action | why this is not append-only | code-size pressure | proof or deferral owner |
|---|---|---|---|---|---|---|
| C1 | B1, B2 | D1 | split | Splits data-plane full vectors from report-plane compact diagnostics in the existing owner. | small add | focused report JSON test |
| C2 | B4 | D2 | rewrite | Rewrites prepared report contents to exclude bulky diagnostics; no new report file. | neutral/reduce output | prepared report size/key test |
| C3 | B3 | D4 | defer | Avoids a generic logging wrapper that could hide useful failure output. | no add | future trigger: another command emits oversized non-actionable stdout |
| C4 | B5 | D3 | keep/extend | Reuses existing test file and fixture style. | small add | pytest |

## 7. Implementation plan

1. Add RED assertions in `tests/test_custom_kg_materialize.py`:
   - report `vector_cache` has `summary`, `missing_counts`, `missing_examples`.
   - report `vector_cache` does not contain `resolved` or raw `vector` arrays.
   - prepared-swap JSON also omits `resolved` and raw vectors.
2. Implement `_compact_vector_cache_report()` in `scripts/custom_kg_incremental.py`.
3. Use the compact projection in `run_full_materialization_no_swap()` report.
4. Keep materialization call using the full in-memory `vector_report["resolved"]`.
5. Verify focused tests, py_compile, and an ad-hoc `/tmp/hermes-verify-*` script that serializes a materialize-full report and checks size/key absence.
6. Sync the tested source script to `/home/xu/project/wiki/lightrag/scripts/custom_kg_incremental.py` and run a non-production-smoke on a tiny temp fixture.

## 8. Proof plan and false-green risks

Proof:

- `test_run_full_materialization_no_swap_fills_true_adds_after_cache_only_blocker` should still pass and additionally prove compact report shape.
- `test_run_full_materialization_prepare_swap_writes_bundle_without_live_swap` should prove prepared swap report is compact.
- Full materialization shadow audit must remain ok.
- Ad-hoc verifier checks serialized JSON contains no `"vector": [` and remains small for a tiny fixture.

False-green risks:

- Checking only returned report but not persisted prepared report would miss bundle bloat.
- Checking only summary but not absence of `resolved` would miss future raw vector leakage.
- Running only compile would miss behavior.

## 9. Blast-radius and rollback plan

Blast radius is limited to `materialize-full` report contents and tests. Storage output, vector cache schema, manifest content, scheduler policy, service stop/start, and import mode selection remain unchanged.

Rollback: revert the compact projection patch if a downstream tool truly requires `vector_cache.resolved` in the report; then reintroduce full vectors only behind explicit debug artifact/mode, not default stdout/log.

## 10. Open questions or deferrals

- Generic subprocess stdout truncation is deferred. The current root cause is upstream report bloat; fixing it preserves error visibility.
- Existing historical huge logs are not pruned by this implementation unless explicitly requested.

## 11. Post-change audit

Implemented rows: D1/D2/D3; D4 remains explicitly deferred.

Actual changes:

- `scripts/custom_kg_incremental.py` adds `compact_vector_cache_report()` and uses it for the `run_full_materialization_no_swap()` report-facing `vector_cache` field.
- The full `vector_report["resolved"]` remains in memory and is still passed to `materialize_file_storage_from_manifest()` before compaction, so storage materialization behavior is unchanged.
- Prepared-swap reports now persist compact `vector_cache` diagnostics with keys `summary`, `missing_counts`, and `missing_examples`; raw `resolved` entries and raw `vector` arrays are omitted from default stdout and prepared reports.
- `tests/test_custom_kg_materialize.py` now asserts the returned report and persisted prepared report are compact and small for the fixture.

RED/GREEN evidence:

- RED: targeted tests failed before implementation because `report["vector_cache"]` had `resolved` and `missing` instead of compact keys.
- GREEN: targeted tests passed after implementation.
- Focused gate: `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -p no:cacheprovider tests/test_custom_kg_materialize.py -q` -> `20 passed`.
- Scheduler-adjacent gate: `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -p no:cacheprovider tests/test_wiki_lightrag_lib.py -k 'batch_lightrag_refresh or prepare_swap or finalize_prepared_swap or select_import_commands or refresh_command_groups' -q` -> `15 passed, 96 deselected`.
- Compile gate: `PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile scripts/custom_kg_incremental.py tests/test_custom_kg_materialize.py` -> passed.
- Broad gate: `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -p no:cacheprovider tests -q` -> `144 passed`.
- Ad-hoc verifier: `/tmp/hermes-verify-thdd3adq.py` exited `0` and was removed; tiny prepare-swap report was 3,759 bytes, returned report was 3,129 bytes, and raw `"vector": [` occurrences were `0` in both.

Diff audit: the patch changes the owning report seam only. It does not change vector cache resolution, storage materialization, scheduler policy, service stop/start, active storage, or manifest semantics.

Live sync evidence:

- `/home/xu/project/wiki/lightrag/scripts/custom_kg_incremental.py` was backed up to `/home/xu/tmp/lightrag_script_backups/20260627_224250_compact_log_fix/custom_kg_incremental.py` and replaced with the tested source version.
- Source/live SHA after sync: `ddad44ef6cf20e026ad225638c5a12ee4209d451378dd4692f5435ee3d22995a`.
- Live py_compile passed, then live `scripts/__pycache__` was removed.
- Live ad-hoc verifier `/tmp/hermes-verify-n70lyfgy.py` exited `0` and was removed; tiny live prepared report was 3,819 bytes with `raw_vector_occurrences=0`.
- Production service health remained `healthy`, pre-query `should_refresh=false`, `pending_count=0`, and `state/prepared_swaps/` was empty.
