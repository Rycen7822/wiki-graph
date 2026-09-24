# Wikigraph/native operations router

Use this for relevance-aware native retrieval over zvec plus its lexical sidecar: status, focused/coverage queries, refresh/materialization, live retrieval smoke tests, response profiles, read-span/reread checks, release-quality reports, service replacement, rollback, state cleanup, section-similarity planning, and refresh bottleneck audits. Production content root is `${LLM_WIKI_ROOT}`; generated native state lives outside the wiki under `${LLM_WIKI_STATE_DIR}`.

## Boundaries

- Production root: `${LLM_WIKI_ROOT}`.
- Workdir/source repo: `${WIKI_GRAPH_REPO}`.
- Ops module root: `${WIKI_GRAPH_REPO}/ops`.
- Central storage root: `${LLM_WIKI_STORAGE_ROOT}`.
- Production native state dir: `${LLM_WIKI_STATE_DIR}`.
- Production native workspace root: `${LLM_WIKI_STATE_DIR}/native_zvec/workspaces`.
- Always pass explicit `--root`, `--state-dir`, `--workdir`, workspace pointer paths, and server URLs. Do not rely on defaults when safety boundaries matter.
- Do not place native SQLite, zvec records, generated JSONL, reports, service logs, scratch, sidecars, or query evidence packs inside the human wiki root.

## Queue/status checks

Production status:

```bash
cd "$WIKI_GRAPH_REPO"
python3 -m ops.batch_wiki_integration status \
  --root "$LLM_WIKI_ROOT" \
  --state-dir "$LLM_WIKI_STATE_DIR"

python3 -m ops.batch_native_refresh status \
  --root "$LLM_WIKI_ROOT" \
  --state-dir "$LLM_WIKI_STATE_DIR" \
  --workdir "$WIKI_GRAPH_REPO"
```

Run wiki integration before native refresh/materialization when raw-fast pending items exist. Do not clear pending ledgers by hand. If production has pending wiki integration, any workspace built from stale state is not graph freshness.

Native graph refresh policy:

- Every successful batch wiki integration must be followed immediately by a native graph refresh when `pending_native_refresh.json` is non-empty.
- `python3 -m ops.batch_native_refresh status ...` exposes `next_refresh_kind`, `completed_incremental_refresh_count`, `incremental_rebuild_threshold`, and `vector_cache_path`.
- Use `incremental` for ordinary graph updates. After 5 completed incremental graph updates, `next_refresh_kind` becomes `full-rebuild`; auto-integration follow-through should consume that due full rebuild in the same guarded workflow before resetting to incremental cycles.
- Every update/rebuild uses vector cache: pass `--fill-missing-vectors` to refresh commands and verify `vector_cache_required=true` / `vector_cache_path` in status or reports.
- Prepare-only output is not live graph freshness. Live freshness requires cutover, service restart, `/health`, `/query/data`, and native pending clear.

### Automated semantic-artifact gate

Normal batch integration owns the repair chain. Before native materialization, `ops.batch_wiki_integration` automatically calls `ops.native_semantic_artifact_refresh`: it rebuilds method atoms, raw sections, and seed edges in parallel; refreshes cached section-similarity artifacts; exports the custom-KG manifest using the explicit workdir `.env`; and compares the just-integrated raw paths against current parsed section content, manifest source coverage, and the vector-cache embedding contract. Native cutover runs only when this gate returns `ok=true`. After cutover it verifies each integrated path by exact `source_path` in active SQLite, including expected section records and lexical spans.

Treat the compact `native_refresh.semantic_artifacts`, `active_workspace_coverage`, and `status_after.should_refresh` fields as the decision surface. Inspect detailed artifacts only for returned failure codes. A pre-cutover gate failure returns code 18 and leaves native pending intact; a post-cutover active-coverage failure returns code 19 and marks native pending again.

Retry a failed post-integration repair — or pick up a deferred native refresh after a `--defer-native-refresh` closeout — with one command; it reruns the semantic gate and guarded native follow-through even when wiki pending was already cleared. `--embedding-profile` overrides the fill profile (default: `LLM_WIKI_NATIVE_EMBEDDING_PROFILE` or `conservative`):

```bash
cd "$WIKI_GRAPH_REPO"
python3 -m ops.batch_wiki_integration refresh-native-after-integration \
  --root "$LLM_WIKI_ROOT" \
  --state-dir "$LLM_WIKI_STATE_DIR" \
  --workdir "$WIKI_GRAPH_REPO"
```

When `LLM_WIKI_NATIVE_LOCAL_EMBEDDING_DOCKER_MANAGED=true`, this owner command starts the configured Compose embedding service, waits for health, and stops plus verifies it on every success or failure path; require `embedding_service.stopped=true`. Only open the report named by `report_path` when `ok=false`; route repairs by the returned failure code (`raw-section-stale`, `custom-kg-source-coverage-missing`, active-workspace coverage missing, manifest/runtime contract mismatch, vector-cache contract change, or managed service codes 20/21). Use `--allow-embedding-contract-change` only for an intentional model/dimension migration. Do not clear either pending ledger by hand.

Vector-cache-backed prepare command:

```bash
python3 -m ops.batch_native_refresh refresh \
  --root "$LLM_WIKI_ROOT" \
  --state-dir "$LLM_WIKI_STATE_DIR" \
  --workdir "$WIKI_GRAPH_REPO" \
  --prepare-only \
  --fill-missing-vectors
```

Guarded live refresh/cutover must also include `--fill-missing-vectors` plus the explicit restart, health, `/query/data`, and unchanged-path guards for the running deployment.

## Production native materialization

`ops.batch_native_refresh` intentionally rejects direct production-root cutover through its higher-level safety path. For production zvec baseline/cutover after explicit authorization, use the lower-level materializer with explicit paths and keep pending ledgers intact unless a real validated integration/refresh completed:

```bash
cd "$WIKI_GRAPH_REPO"
python3 -m ops.native_zvec_materialize preflight \
  --root "$LLM_WIKI_ROOT" \
  --state-dir "$LLM_WIKI_STATE_DIR" \
  --workspace-root "${LLM_WIKI_STATE_DIR}/native_zvec/workspaces" \
  --workspace-id native-prod-<stamp>

python3 -m ops.native_zvec_materialize build \
  --prepare-only \
  --root "$LLM_WIKI_ROOT" \
  --state-dir "$LLM_WIKI_STATE_DIR" \
  --workspace-root "${LLM_WIKI_STATE_DIR}/native_zvec/workspaces" \
  --workspace-id native-prod-<stamp>
```

Finalize only after build/audit passes and the intended service switch/rollback plan is clear:

```bash
cd "$WIKI_GRAPH_REPO"
python3 -m ops.native_zvec_materialize finalize \
  --workspace-root "${LLM_WIKI_STATE_DIR}/native_zvec/workspaces" \
  --reason '<reason>'
```

## Native API service and retrieval smoke

Production service restart uses the repo-owned native server control command. Runtime configuration lives in `${WIKI_GRAPH_REPO}/.env`; the command resolves the wiki repo/root, native state dir, workspace pointer, log dir, pidfile, host/port, health URL, and server URL from that file.

Reusable restart command:

```bash
cd "$WIKI_GRAPH_REPO"
python3 -m ops.native_server_control restart
```

Before a live restart, inspect the resolved command without touching the process:

```bash
cd "$WIKI_GRAPH_REPO"
python3 -m ops.native_server_control restart --dry-run
```

The script loads `.env`, stops the pid recorded in `LLM_WIKI_NATIVE_SERVER_PIDFILE`, starts `llm_wiki_native.api.server` with `.env` paths, writes logs under `LLM_WIKI_NATIVE_SERVER_LOG_DIR`, waits for `LLM_WIKI_NATIVE_HEALTH_URL`, and reports JSON. Use `python3 -m ops.native_server_control config` when only path/command inspection is needed.

`/query` may run in answer-provider bypass mode when no external answer LLM is authorized. `/query/data` is the required retrieval proof surface. If embedding env is not configured or production query text should not be sent to an embedding endpoint, pass an explicit `query_vector` from the active SQLite workspace.

Native retrieval sends zvec navigation/section and lexical candidates through one `relevance-v1` planner. Choose the goal explicitly when intent matters:

- `retrieval_goal="focused"` is the default and favors the smallest decisive evidence set while suppressing redundant evidence.
- `retrieval_goal="coverage"` fills distinct sources for comparisons, surveys, evidence/limitation coverage, and other multi-source questions.

`top_k` is the maximum number of visible context blocks, not the number of initially retrieved candidates. Candidate limits are bounded internal implementation details, not public tuning knobs.

Use response profiles to control output size:

- `response_profile="compact"`: minimal context blocks; omits neighbor/ranking debug bulk while preserving `coverage_plan`, source paths, and read-span cards.
- `response_profile="standard"`: default context detail for normal retrieval evidence.
- `response_profile="debug"`: includes planner decisions, route ranks, relevance breakdowns, and source scope for retrieval audits.

Focused `/query/data` retrieval:

```bash
cd "$WIKI_GRAPH_REPO"
python3 -m ops.wiki_search "<query>" \
  --server "$LLM_WIKI_SERVER" \
  --retrieval-goal focused \
  --top-k 4 \
  --response-profile standard \
  --data-only \
  --no-record-query-event
```

Use `--retrieval-goal coverage --top-k 6` for multi-source comparison or coverage work. Add `--query-vector '[...]'` when query embedding must remain local/precomputed.

Responses with `ranking_contract="relevance-v1"` expose `score` as the weighted final relevance score. Use `relevance_score_breakdown` and `route_ranks` to interpret it; legacy `score_breakdown`, debug hits, and coverage fields remain additive compatibility data and must not be used to reconstruct the final score. Scores remain retrieval evidence, not source truth: reread the selected raw/source files before making factual claims.

Lexical sidecar rows cover document headings, `_meta` map rows, Markdown table rows, and state-only fallback spans. Existing active workspaces built before the sidecar change remain valid but may have empty lexical-span tables until a new prepared workspace is built and finalized. Do not claim table-row/map-row coverage or read-span availability unless `/query/data` trace/debug output or SQLite sidecar counts prove lexical hits are present.

Use `/read/span` only for sidecar span ids returned in context `read_span` cards or debug hits. With `source_root` in the active pointer, it rereads the current source file and relocates moved text by exact match; without source root it returns stored snapshot text. Missing/stale status means the span cannot be treated as current-source evidence until the file is reopened manually or the workspace is rebuilt.

Example read-span request:

```bash
curl -sS "$LLM_WIKI_NATIVE_SERVER_URL/read/span" \
  -H 'content-type: application/json' \
  -d '{"workspace_id":"<active-workspace-id>","span_id":"<span-id>"}'
```

For candidate or active release validation, use `ops.collect_native_query_report` with `--quality-contract relevance-v1`, a frozen audited workspace/suite and matching baseline, `--partition all`, `--runtime-code-root`, `--require-gates`, and `--fail-if-output-exists`. Active validation additionally supplies `--accepted-candidate-report`; use `--promote-on-pass` for the canonical destination and do not promote a report unless the command returns zero. Keep reports, large debug profiles, and evidence packs outside the human wiki root.

A planner/runtime-only release does not itself require a workspace rebuild, native refresh, or pointer cutover. Corpus, embedding, schema, or index changes remain separate refresh/cutover workflows.

## Section-similarity graph and query expansion

Treat embedding-derived edges as section-level semantic-neighbor augmentation, not source truth. Deterministic edges such as `RAW_SECTION_OF`, `SOURCED_BY`, `WIKILINKS_TO`, method-atom edges, and raw/compiled provenance remain primary evidence. Use conservative relation names such as `SEMANTIC_SECTION_NEIGHBOR`; never read them as `SOLVES`, `CAUSES`, `EXTENDS`, or `INSPIRES` without later evidence synthesis.

Current section kinds include summary, abstract, motivation, methodology, results, future, limitations, and questions. Broad summary/abstract/motivation pairs may remain sidecar-only; import sparse high-value pairs such as methodology/results/future/limitations/questions. Formula/objective evidence belongs in methodology sections; figure/table evidence belongs in results/methodology/limitations, not standalone formula/figure sections.

Before enabling a fast section-rank path, prove scalar-vs-fast parity on threshold-adjacent values, deterministic tie-breaks, same-note masking, zero vectors, mismatched dimensions, output byte/JSON parity, ranks, rounded cosines, mutual-kNN flags, pair kinds, and order. Benchmark on copied temp state, then compare candidate JSONL/hash on a representative real-state subset before full-state work. Do not use ANN, lower thresholds, reduce section kinds, or prune relationships unless the user explicitly approves a quality-changing fast profile.

Query expansion should separate direct vector hits, semantic-neighbor expansions, graph-only neighbors, source raw paths, and section titles. Always trace back to original raw notes/sections before making claims; do not cite generated similarity edges as sources or mix embeddings from different models/extraction versions without rebuild gates.

## Read-only refresh bottleneck audit

For audits focused on data volume, section-similarity cost, or graph import cost, stay read-only unless explicitly asked: do not run refresh, clear ledgers, auto-integrate, restart services, or mutate wiki/native state. Safe reads include latest native refresh logs/reports, manifest/build reports, section-similarity reports, raw-section JSONL, section-similarity edge/vector artifacts, pending ledgers, and workspace sizes/counts. Put audit artifacts outside human wiki root and native runtime state unless durable reporting is requested.

Measure artifact generation separately from import wall time. In `custom_kg_import_report.json`, inspect `timings.total_s`, `timings.apply_patch_to_shadow_s`, payload/manifest counts, and diff add/update/delete counts. Classify relationship diffs by semantic keyword using the current manifest before blaming section similarity; deterministic `SOURCED_BY`/`WIKILINKS_TO` waves and relationship vector flush batch size may dominate import time. Count storage volume separately from semantic graph volume: relationship/chunk/entity vector stores, text chunks, GraphML, manifest, and section embeddings.

Preferred bottleneck wording: “Current refresh is dominated by incremental import patching of JSON-backed graph/vector storage, especially relationship vector updates; section similarity is a future scaling risk and contributes artifact/state volume, but was not the latest dominant wall-clock cost.” Avoid “section similarity is the bottleneck” unless latest logs prove it or the question is explicitly about future section-sim scaling.

Engineering options to discuss, not apply automatically: split relationship semantic-content hash from bookkeeping/provenance hash; stabilize deterministic relationship descriptions; shadow-test larger embedding batches/concurrency; add delta section-similarity maintenance with deletion/changed-text/tie/order parity; consider binary/mmap section embedding cache; treat pair-family pruning/raw-section compaction as retrieval-quality tradeoffs.

## Rollback

Before cutover, record current active workspace pointer, service PID/port, and restart command. Rollback is incomplete unless pointer restoration, service restart, `/health`, and `/query/data` all pass. Retain failed workspaces and logs until rollback evidence is clean.

## Reporting

Report compact fields: root, state dir, workspace id, active pointer, port/PID/session, wiki pending/actionable count, native pending count, health status, `/query/data` hit count, watched paths unchanged/changed, pending clear result, rollback result, bottleneck classification, and artifact paths. Keep large JSON/log files as artifacts rather than pasting them into worknotes.
