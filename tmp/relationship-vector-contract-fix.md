# Relationship Vector Contract Fix Implementation Plan

> **For Hermes:** Execute directly in small TDD slices; do not dispatch subagents unless the scope grows beyond this artifact.

**Goal:** Make full materialization/threshold refresh preserve typed relationship semantics while avoiding false near-full relationship re-embedding caused by legacy LightRAG pair-order vector text.

**Architecture:** Keep typed/directed relationship identity (`src<SEP>tgt<SEP>keywords`) as the manifest/VDB owner contract, but normalize relationship embedding text to LightRAG's pair-sorted endpoint content so legacy storage vectors are reusable when the text is exactly identical. Add a narrow embedding-fill gate for the remaining true misses instead of falling back to upstream cold import, which collapses same-endpoint typed edges.

**Design audit:** This file is the design artifact. The change rewrites the existing manifest/vector-cache owner seams instead of adding a parallel refresh wrapper; cold upstream import remains legacy and must not be allowed to write a false typed manifest.

**Tech Stack:** Python 3.12 scripts, pytest, LightRAG file-backend JSON/GraphML/NanoVectorDB storage, OpenAI-compatible embedding endpoint configured from existing `.env` variables.

---

## Request normalization

The user asked to fix the root cause long-term because 8,810 relationship vector misses are abnormal and could break automatic threshold refresh. Constraints: preserve typed relationship semantics, avoid unnecessary re-embedding, keep live service/storage safe, prove with non-mutating production dry-runs before any swap, and do not patch live storage directly.

Mode: design-plus-implementation.

## Sources inspected

- `scripts/custom_kg_incremental.py` lines 120-224, 348-567, 692-929, 1470-1606, 1726-1812.
- `scripts/vector_cache.py` lines 341-431.
- `scripts/custom_kg_materialize.py` lines 54-131.
- `scripts/import_custom_kg.py` lines 123-175.
- `scripts/batch_lightrag_refresh.py` lines 45-170 and 170-257.
- `tests/test_custom_kg_materialize.py` lines 1-439.
- `tests/test_vector_cache.py` lines 1-275.
- Live read-only measurements from `/home/xu/project/wiki/lightrag`: current manifest/storage/cache counts, exact-content storage reuse, and threshold dry-run command selection.

## Design iteration logs

### Iteration 1 — identify false vector churn

1. Question: are the 8,810 misses real semantic vector changes? Answer: no. If relationship vector text uses pair-sorted endpoints, live storage exact-content reuse rises to 18,244/18,492 relationships; the large miss set is dominated by endpoint-order text drift (`RAW_SECTION_OF`, `SEMANTIC_SECTION_NEIGHBOR`).
2. Question: what remains unrecoverable without embedding? Answer: 102 `WIKILINKS_TO` rows where upstream cold import collapsed multiple typed relationships sharing one endpoint pair or retained a different same-pair relation; no exact storage vector was available before the embedding-fill run.

Research added: in-memory alternate-manifest simulations against live `vdb_relationships.json` and `state/vector_cache.sqlite`.
Design change: relationship vector content should be pair-sorted while identity remains typed; add explicit embedding-fill for true misses.
Remaining uncertainty: actual embedding endpoint availability for a real fill run; code can be tested with injected fake embedder.

### Iteration 2 — choose owner seam

1. Question: should this be fixed in batch orchestration or manifest/cache owners? Answer: manifest/cache owners. The wrong contract originates in `build_custom_kg_manifest()` and `seed_vector_cache_from_storage()`, while batch only selects commands.
2. Question: should cold import continue writing successful typed manifests? Answer: no. Upstream `ainsert_custom_kg` dedupes by sorted endpoint pair and cannot prove typed storage parity; add post-import audit before manifest write.
3. Question: how should threshold refresh complete? Answer: `materialize-full` should seed exact legacy vectors, fill only unresolved vectors via the embedding endpoint when explicitly enabled, then write an audited prepared swap.

4. Question: how can a prepared full materialization repair a known-bad live storage? Answer: add an explicit `--allow-current-storage-audit-failure` flag to prepare/finalize paths; default remains fail-closed, but the full+reuse repair path can replace old bad storage after manifest-hash checks and shadow audit pass.

Research added: source reads of materializer, vector cache seeding, batch command builder, and `import_custom_kg.py` manifest write path.
Design change: add `--fill-missing-vectors` to materialize-full and include it in full materialization command selection; keep fail-closed when fill is disabled/unavailable.
Remaining uncertainty: resolved during implementation; true live fill count was 102 after exact-content storage seeding and current-cache reuse.

## Baseline design inventory

| id | existing element | current assumption/contract | evidence | owner/seam | risk if changed |
|---|---|---|---|---|---|
| B1 | Relationship manifest identity | Manifest relationship key/VDB id is typed and directed (`src<SEP>tgt<SEP>keywords`). | `relationship_record_key()` and materialization tests preserve same-endpoint typed rows. | `scripts/custom_kg_incremental.py` manifest builder/materializer. | Collapsing back to pairs loses typed semantics and raw-section/source relation distinctions. |
| B2 | Relationship vector text | Current manifest vector text uses typed direction order (`keywords\tsrc\ntgt\ndescription`). | `build_custom_kg_manifest()` line 472. | Manifest vector hash owner. | Endpoint-order-only changes look like vector changes and block no-reembed full materialization. |
| B3 | Legacy live relationship storage | Upstream LightRAG cold import stores relationships by sorted endpoint pair and embeds `keywords\tsorted_src\nsorted_tgt\ndescription`. | Upstream `ainsert_custom_kg`; live simulation exact matches 18,244 pair-sorted texts. | External LightRAG import path. | Treating this as typed storage writes false successful manifests. |
| B4 | Vector cache seeding | Seed requires previous manifest vector hash equality before using storage vectors. | `seed_vector_cache_from_storage()` lines 379-387. | `scripts/vector_cache.py`. | Safe but too literal for legacy manifest-text normalization; it rejects storage vectors whose stored content exactly matches the new safe text. |
| B5 | Full materialization missing-vector behavior | Materialize-full requires all vectors pre-resolved and otherwise fails. | `run_full_materialization_no_swap()` lines 1511-1515. | `scripts/custom_kg_incremental.py`. | Threshold full refresh cannot complete when there are true new typed vectors. |
| B6 | Cold import manifest write | `import_custom_kg.py` writes desired manifest after upstream import without auditing storage parity. | `import_custom_kg.py` lines 159-170. | Cold import script. | Recreates typed-manifest/live-storage mismatch after every cold import. |
| B7 | Batch threshold full materialization | Threshold/default forced full rebuild uses materialize-full with seed-from-storage. | current uncommitted `batch_lightrag_refresh.py` and dry-run. | Batch orchestration. | Needs fill flag to complete true deltas; must not silently cold fallback. |
| B8 | Prepared swap live pre-audit | Prepare/finalize historically require current live storage audit to pass before repair. | Real forced full+reuse failed before service stop while old live storage was known-bad. | `run_full_materialization_no_swap()`, `run_finalize_prepared_swap()`. | Prevents repairing exactly the old storage mismatch that the prepared shadow is meant to replace. |

## Proposed design ledger

| id | baseline refs | proposed decision | intent | files/seams touched | expected impact | rollback/proof |
|---|---|---|---|---|---|---|
| D1 | B1,B2,B3 | Rewrite relationship vector content helper to use pair-sorted endpoints while preserving typed record keys/VDB ids. | Remove false endpoint-order vector churn without losing typed identity. | `scripts/custom_kg_incremental.py`, tests. | 8,810 current misses shrink to true missing typed rows. | Regression proves reversed endpoint relationship hashes match legacy sorted content, while typed VDB ids remain distinct. |
| D2 | B4 | Allow storage seeding across a previous-manifest vector-hash mismatch only when the storage record content exactly matches desired content and embedding contract is unchanged. | Reuse real legacy vectors safely after text-normalization migration. | `scripts/vector_cache.py`, tests. | Seed 18,244 relationship vectors from live storage instead of zero. | Regression proves embedding contract changes still skip. |
| D3 | B5,B7 | Add materialize-full embedding-fill for unresolved vectors behind `--fill-missing-vectors`; batch full materialization includes it so threshold can finish true deltas. | Fill only the true missing typed vectors, not the false 8,810. | `scripts/custom_kg_incremental.py`, `scripts/batch_lightrag_refresh.py`, tests. | Threshold refresh becomes self-healing when embedding endpoint is configured. | Fake-embed tests prove only missing records are embedded and cache resolves; no network in tests. |
| D4 | B6 | Add post-cold-import audit before writing successful manifest/report. | Prevent future false-green typed manifests from upstream pair-collapsed cold imports. | `scripts/import_custom_kg.py`, tests if practical. | Cold import fails closed unless storage matches manifest; materialize-full becomes preferred. | Dry-run unaffected; test or compile gate; rollback by reverting audit addition only. |
| D5 | B8 | Add explicit `--allow-current-storage-audit-failure` to prepare/finalize repair path and include it only in full-materialization commands. | Let an audited prepared shadow replace known-bad live storage without weakening default pre-audit guards. | `scripts/custom_kg_incremental.py`, `scripts/batch_lightrag_refresh.py`, tests. | Real full+reuse repair can complete; ordinary prepared swaps still fail closed by default. | RED/GREEN tests prove default blocker and explicit repair path; final live audit must be clean after swap. |

## Compression review

| id | baseline refs | decision refs | compression action | why this is not append-only | code-size pressure | proof or deferral owner |
|---|---|---|---|---|---|---|
| C1 | B1,B2,B3 | D1 | rewrite | Changes the manifest vector-text owner instead of adding a compatibility wrapper around every refresh. | neutral | `test_*relationship_vector_content*` and production dry-run counts. |
| C2 | B4 | D2 | rewrite | Extends the existing seed seam with exact-content proof instead of copying live vectors in a separate repair script. | small add | `tests/test_vector_cache.py` RED/GREEN. |
| C3 | B5,B7 | D3 | split | Separates cache reuse from true embedding fill; avoids cold import fallback and avoids pretending missing vectors are reusable. | moderate add | `tests/test_custom_kg_materialize.py` fake embedder plus dry-run command check. |
| C4 | B6 | D4 | rewrite | Makes the existing cold import honest; no new cold-import mode. | small add | compile and targeted import tests/dry-run. |
| C5 | B8 | D5 | narrow flag | Keeps normal live pre-audit fail-closed and adds only an explicit repair-mode override for known-bad live storage. | small add | RED/GREEN prepare/finalize tests plus successful live post-swap audit. |

## Implementation plan

### Task 1: Relationship vector text helper

Design refs: B1-B3, D1, C1.

1. Add `relationship_vector_content(src_id, tgt_id, keywords, description)` in `scripts/custom_kg_incremental.py`; it sorts endpoints with the same ordering as `relation_chunk_key()`.
2. Replace inline relationship `content` construction in `build_custom_kg_manifest()` with the helper.
3. Update the stale docstring that says relationships keep only the last unordered endpoint pair.
4. Add a focused test proving two typed same-pair relationships keep distinct VDB ids while vector text endpoint lines are pair-sorted.
5. Run the targeted test and `py_compile`.

### Task 2: Exact-content legacy seed compatibility

Design refs: B4, D2, C2.

1. Add a small helper in `scripts/vector_cache.py` that permits previous-vector mismatch only when collection is `relationships`, embedding contract is unchanged, storage content equals desired `content`, and vector decodes to the desired dimension.
2. Preserve the existing mismatch skip for chunks/entities and for real embedding-model/dimension/params changes.
3. Add RED tests in `tests/test_vector_cache.py`: one legacy relationship sorted-content seed should pass; one embedding-contract change should still skip.
4. Run targeted vector cache tests.

### Task 3: Fill true missing vectors explicitly

Design refs: B5,B7, D3, C3.

1. Add `fill_missing_manifest_vectors()` in `scripts/custom_kg_incremental.py` with injectable embed function for tests.
2. Use existing `.env` names: `EMBEDDING_BINDING_HOST`/`OPENAI_BASE_URL`, `EMBEDDING_BINDING_API_KEY`/`OPENAI_API_KEY`, `EMBEDDING_MODEL`, `EMBEDDING_DIM`, `EMBEDDING_BATCH_NUM`, `EMBEDDING_TIMEOUT`.
3. In `run_full_materialization_no_swap()`, after seed+resolve, if misses remain and `--fill-missing-vectors` is set, embed only missing records and resolve again; if misses remain, fail closed.
4. Add CLI flag `--fill-missing-vectors`.
5. Update batch full materialization command to pass the flag for threshold/full materialization path.
6. Add fake-embed test proving only missing records are embedded and materialization succeeds.

### Task 4: Cold import honesty gate

Design refs: B6, D4, C4.

1. After `rag.finalize_storages()` in `scripts/import_custom_kg.py`, run `audit_custom_kg_storage(workdir / "rag_storage", desired_manifest)`.
2. If audit fails, write the audit sample into summary and raise before `write_successful_manifest()`.
3. Keep `--dry-run` behavior unchanged.
4. Run compile and targeted tests.

### Task 4.5: Prepared-swap repair gate for known-bad live storage

Design refs: B8, D5, C5.

1. Add `--allow-current-storage-audit-failure` to `materialize-full --prepare-swap` and `finalize-prepared-swap`.
2. Keep default behavior fail-closed when the flag is absent.
3. Include the flag in full-materialization batch commands only, not in ordinary incremental prepared swaps.
4. Add RED/GREEN tests proving prepare/finalize can repair known-bad live storage only under the explicit flag.
5. Run the real forced full+reuse refresh and require final post-swap audit to pass.

### Task 5: Production non-mutating proof and live sync

Design refs: all.

1. Run targeted pytest and py_compile in `wiki-graph`.
2. Run read-only production simulation/dry-run from `lightrag`: expected relationship misses collapse from 8,810 to the small true-missing set before fill, and command includes `--fill-missing-vectors` for threshold/full materialization.
3. Sync touched scripts/tests from `wiki-graph` to `lightrag` only after tests pass.
4. Run live `materialize-full --no-swap --seed-from-storage --fill-missing-vectors --delete-shadow-on-no-swap` only if embedding endpoint is configured and the user-authorized fix scope still includes applying the refresh; otherwise stop after dry-run and report that code is ready but true missing rows require an embedding-fill run.
5. Verify service health remains healthy and no live swap happened unless explicitly authorized by the command path.

## Proof plan and false-green risks

- RED/GREEN tests must prove legacy endpoint-order normalization is not treated as vector churn.
- False-green risk: reusing vectors after embedding model/dim/params change. Guard with explicit contract mismatch test.
- False-green risk: same endpoint pair with multiple typed relationships. Guard by preserving typed keys/VDB ids and requiring embedding-fill for rows not exactly present in storage/cache.
- False-green risk: cold import writes false manifest. Guard with post-import audit.
- Production proof must be non-mutating until tests pass: dry-run, temp cache simulation, health check.

## Blast-radius and rollback plan

Touched source is limited to `scripts/custom_kg_incremental.py`, `scripts/vector_cache.py`, `scripts/batch_lightrag_refresh.py`, `scripts/import_custom_kg.py`, and focused tests. Rollback is a normal git revert of those files; live storage is not modified by code tests/dry-runs. If a real embedding-fill materialization is run and fails before finalize, no live swap occurs because it uses prepared/no-swap guards.

## Open questions or deferrals

- Whether to immediately finalize a prepared full materialization into live storage after code passes remains a separate operator decision unless the command is explicitly executed and health/audit gates pass.
- The 102 true missing typed relationship vectors required one embedding-fill run; they were not hidden by aliasing a different relation's vector.

## Final validation record

- Staging focused owner suites: `tests/test_custom_kg_materialize.py tests/test_vector_cache.py` -> `30 passed`.
- Staging `tests/test_wiki_lightrag_lib.py` -> `111 passed`.
- Staging full root suite: `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -p no:cacheprovider tests -q` -> `143 passed`.
- Live targeted tests after final sync: repair/CLI tests -> `3 passed`; batch dry-run tests -> `2 passed`; py_compile exit 0.
- Live forced full+reuse refresh succeeded and finalized prepared swap: `import_mode=full_materialization`, `swapped=true`, `finalized_prepared_swap=true`.
- Live post-swap audit: `ok=true`, issues `[]`, graph nodes 9,592, graph edges 18,244, vdb relationships 18,492, relation chunks 18,492, isolates 0.
- Live vector cache now resolves all 37,636 desired vectors, including all 18,492 relationships.
- Live refresh ledger: `dirty=false`, pending count 0, `last_successful_refresh_at=2026-06-27 16:15:18`.
- Live service health after cleanup: `healthy`, `pipeline_busy=false`, `pipeline_active=false`, `pipeline_destructive_busy=false`.
