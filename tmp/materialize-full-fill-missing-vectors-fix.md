# Materialize-full fill-missing-vectors fix design audit

## 1. Request normalization

User asked to implement the recommended no-extra-external-control-cost fix for the LightRAG threshold full-materialization failure. Desired outcome: `materialize-full --fill-missing-vectors` must internally handle true新增/vector_update records after cache/storage seeding instead of failing early with the cache-only blocker. Non-goal: do not add closeout/cron/manual fallback logic, do not change threshold dispatch back to incremental, do not cold-reembed the whole graph, and do not mutate live runtime until source tests pass.

Mode: design-plus-implementation.

## 2. Sources inspected

- `scripts/custom_kg_incremental.py:607-628` — `full_materialization_cache_only_blockers()` treats adds/vector updates as unsafe for cache-only materialization.
- `scripts/custom_kg_incremental.py:680-742` — `fill_missing_manifest_vectors()` already embeds only unresolved manifest vectors and writes them to `VectorCache`.
- `scripts/custom_kg_incremental.py:1636-1706` — `run_full_materialization_no_swap()` computes blockers before seed/resolve/fill and always raises when blockers exist.
- `scripts/batch_lightrag_refresh.py:133-173` — full-materialization command already passes `--fill-missing-vectors` and prepared-swap flags.
- `tests/test_custom_kg_materialize.py:258-358` — existing tests cover cache-only blocking and missing-vector fill, but not the combined previous-manifest + add/vector_update + fill-enabled path that failed in production.
- `tests/test_wiki_lightrag_lib.py:687-749` — threshold dry-run proves threshold defaults to full+reuse and passes `--fill-missing-vectors`.
- Runtime evidence from `state/refresh_logs/20260627_184647_batch_lightrag_refresh_import.log` — threshold full+reuse failed before service stop with `new_or_vector_updated_records=506` despite `--fill-missing-vectors`.

## 3. Design iteration logs

### Iteration 1 — root cause and owner seam

1. Question: Is failure caused by scheduler fallback policy or materializer behavior? Answer: materializer behavior. The scheduler already supplies `--fill-missing-vectors`; the materializer rejects before it can execute the existing fill step.
2. Question: Should the fix live in closeout/runner fallback? Answer: no. That would add external control cost and duplicate policy in callers.

Research added: source lines in `custom_kg_incremental.py` and `batch_lightrag_refresh.py`; production failure log.
Design change: fix owning materializer seam rather than adding caller fallback.
Remaining uncertainty: exact report field shape for blocker diagnostics.

### Iteration 2 — proof and compatibility

1. Question: Can existing fill path cover true misses? Answer: yes. `fill_missing_manifest_vectors()` derives missing records from `resolve_manifest_vectors()`, embeds only unresolved records, validates model/dim/contract, writes to cache, and then current code resolves again before materialization.
2. Question: What compatibility must remain? Answer: cache-only runs without `--fill-missing-vectors` must still fail early on true adds/vector updates; final unresolved misses must still fail closed before service stop.

Research added: existing tests around blocker and fill path.
Design change: blocker becomes conditional on `not fill_missing_vectors`; report includes blocker summary regardless.
Remaining uncertainty: none for source-level behavior; live endpoint availability remains an implementation-time runtime dependency if real fill is needed.

## 4. Baseline design inventory

| id | existing element | current assumption/contract | evidence | owner/seam | risk if changed |
|---|---|---|---|---|---|
| B1 | `full_materialization_cache_only_blockers()` | Adds/vector updates are unsafe for cache-only assembly. | `custom_kg_incremental.py:607-628` | manifest diff safety seam | If removed entirely, cache-only materialization could falsely green without embeddings. |
| B2 | `run_full_materialization_no_swap()` blocker placement | Block before cache seed/resolve/fill. | `custom_kg_incremental.py:1668-1674` | full materializer lifecycle | Blocks legitimate `--fill-missing-vectors` runs. |
| B3 | `fill_missing_manifest_vectors()` | Existing internal fill can embed only unresolved records after cache resolution. | `custom_kg_incremental.py:680-742` | vector-cache fill seam | If bypassed, caller must manage fallback externally. |
| B4 | threshold scheduler | Threshold refresh force-selects full+reuse and passes `--fill-missing-vectors`. | `batch_lightrag_refresh.py:133-173`, tests | batch refresh entrypoint | Changing this would weaken full+prepared-swap goal and spread control policy. |
| B5 | fail-closed prepared-swap rule | Explicit/threshold full+reuse must not silently cold fallback. | llm-wiki M1 reference | runtime safety contract | Silent fallback risks expensive re-embedding or unsafe service stop behavior. |

## 5. Proposed design ledger

| id | baseline refs | proposed decision | intent | files/seams touched | expected impact | rollback/proof |
|---|---|---|---|---|---|---|
| D1 | B1,B2,B3 | In `run_full_materialization_no_swap()`, raise cache-only blocker only when blockers exist and `fill_missing_vectors` is false. | Make `--fill-missing-vectors` the internal, no-extra-control-cost path for true misses. | `scripts/custom_kg_incremental.py` | Threshold full+reuse can fill true adds/vector updates internally. | Existing cache-only blocker test plus new fill-enabled regression. |
| D2 | B1,B2 | Keep blocker diagnostic in report even when fill is enabled. | Preserve production observability without changing control flow. | `scripts/custom_kg_incremental.py` report dict | Future logs show what was filled/why. | Test asserts report has `cache_only_blockers` while succeeding. |
| D3 | B4,B5 | Do not change scheduler fallback or threshold mode selection. | Avoid external control cost and preserve prepared-swap architecture. | no scheduler behavior change | Existing threshold tests remain compatibility gates. | `test_wiki_lightrag_lib` focused gates. |

## 6. Compression review

| id | baseline refs | decision refs | compression action | why this is not append-only | code-size pressure | proof or deferral owner |
|---|---|---|---|---|---|---|
| C1 | B2 | D1 | rewrite | Changes the stale early guard in the owning lifecycle instead of adding a new fallback wrapper. | neutral/minimal | `tests/test_custom_kg_materialize.py` RED/GREEN. |
| C2 | B4,B5 | D3 | keep | Scheduler already supplies correct flags; no need for new policy branch. | no added code | existing scheduler tests. |
| C3 | B1 | D2 | keep+narrow | Preserve blocker semantics for cache-only, narrow only when fill is explicitly enabled. | small report-field addition | report assertion in new regression. |

## 7. Implementation plan

1. Add failing regression in `tests/test_custom_kg_materialize.py` using previous manifest + desired added entity, an incomplete cache, and `fill_missing_vectors=True`; expect success, one embedded new entity, zero misses, and `cache_only_blockers.blocked=true` recorded.
2. Run that test and verify it fails with current `cache-only full materialization is unsafe` error.
3. Patch `scripts/custom_kg_incremental.py` to compute `fill_missing_vectors_enabled`, gate early blocker only when false, update error text to mention `--fill-missing-vectors`, and include `cache_only_blockers` in the returned report.
4. Run focused custom materialize tests, scheduler tests, py_compile, and full test suite if focused gates pass.
5. Sync tested source scripts to `/home/xu/project/wiki/lightrag/scripts/` only after source tests pass; run final `validate_wiki.py --full --write-report` and refresh status checks before reporting graph freshness.

## 8. Proof plan and false-green risks

- RED proof: new regression fails before code change because the early blocker fires before fill.
- GREEN proof: new regression succeeds and confirms only the missing new vector was embedded.
- Compatibility proof: existing cache-only blocker test still fails closed without `fill_missing_vectors`.
- Scheduler compatibility: threshold/manual full+reuse dry-runs still include `--fill-missing-vectors` and avoid cold import.
- False-green risk: test cache accidentally includes every desired vector; avoid by seeding only previous manifest vectors and asserting the embedder sees the new record content.
- False-green risk: report claims success before final resolve; assert zero final misses and shadow audit ok.

## 9. Blast-radius and rollback plan

Blast radius is limited to `materialize-full` no-swap/prepared-swap behavior and tests. Runtime sync touches only script files after source validation. Rollback is a single-file revert of `scripts/custom_kg_incremental.py` plus test revert; live rollback is copying the previous deployed script from git checkout or reverting sync.

## 10. Open questions or deferrals

- No new external fallback policy is introduced.
- No zvec/storage backend change is included.
- No log-size patch is included unless separately requested.

## 11. Post-change audit

Implemented rows: B1/B2/B3 via D1/D2 and C1/C3; B4/B5 protected via D3/C2 with no scheduler behavior change.

Actual source changes:

- `scripts/custom_kg_incremental.py` now computes `fill_missing_vectors_enabled`, raises the cache-only blocker only when blockers exist and fill is disabled, reuses that flag for the existing fill step, and records a compact `cache_only_blockers` summary plus `fill_missing_vectors` in the materialization report.
- `tests/test_custom_kg_materialize.py` now has a regression where a previous manifest plus a desired added entity succeeds with `fill_missing_vectors=True`, embeds only the new entity, records compact blockers, and audits the shadow.

RED/GREEN evidence:

- RED before source change: `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -p no:cacheprovider tests/test_custom_kg_materialize.py::test_run_full_materialization_no_swap_fills_true_adds_after_cache_only_blocker -q` failed with `cache-only full materialization is unsafe ... new_or_vector_updated_records=1`.
- GREEN after source change: the same regression plus adjacent blocker/fill tests passed.
- Focused gates passed: `tests/test_custom_kg_materialize.py -q` -> `20 passed`; `tests/test_wiki_lightrag_lib.py -k 'batch_lightrag_refresh or prepare_swap or finalize_prepared_swap or select_import_commands or refresh_command_groups' -q` -> `15 passed, 96 deselected`; `py_compile` passed.
- Broad source gate passed: `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -p no:cacheprovider tests -q` -> `144 passed`.

Diff audit: the implementation changed the owning materializer seam instead of adding an external fallback or changing threshold dispatch. The report field is compact and excludes the full diff/id arrays to avoid log bloat.
