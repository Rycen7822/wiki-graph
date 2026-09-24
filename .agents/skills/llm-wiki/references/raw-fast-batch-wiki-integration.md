# Raw-fast batch wiki integration

Use this for default clipping when the immediate deliverable is a verified canonical raw note, including raw-note-only manifests, checklist/DONE batches, and post-compression closeout reconciliation. Raw-fast means: save raw now, verify it, mark wiki integration pending, and defer compiled/meta/log/full graph work until threshold, explicit batch integration, or pre-query integration.

## Core contract

1. Orient just enough to avoid duplicates and schema violations.
2. Preferred evidence setup: run `python3 -m ops.raw_fast_ingest_prepare --url <source-url>`. The production profile supplies the wiki root, state dir, and per-task tmp root; pass explicit `--root`, `--state-dir`, `--workdir`, or `--tmp-root` only for tests, repair, or non-production runs. The wrapper writes only temp evidence, returns `agent_next_reads`, prepares source-exposed resource triage plus closeout args, resolves known arXiv-discovery routes such as Hugging Face paper pages, PapersWithCode numeric paper pages, alphaXiv numeric pages, and DAIR academy paper pages to canonical arXiv source reads, and chooses TeX/source first for usable arXiv e-prints with Docling as the direct-PDF or TeX-missing fallback.
3. Read `agent_handoff.md` first, then follow its `source_read_plan`. When `resource_review_required=false`, stay with generated source reads and use only the sanitized `paper_digest.md` scientific digest for a specific scientific gap; do not open internal candidate/evidence/brief sidecars on the happy path.
4. Manual repair is a returned-output branch: when prepare/handoff returns `manual_required=true`, `resource_review_required=true`, or `manual_reference_paths`, read exactly those paths and follow the concrete failure reason.
5. Write the handoff's `raw_body_draft.md` as body-only scientific reading prose. Run the generated `assemble_command.preview.sh` to create the canonical raw note under `raw/clip/<YYMM>/...md` from script-owned `candidate_frontmatter.json` plus the body draft. `<YYMM>` and `YYMMDDNN` come from clipping/raw-note creation time. The prepare-time `next_raw_path` is provisional; `ops.raw_fast_publish` allocates the final path under the shared mutation lock and rewrites generated closeout artifacts if a collision changes it.
6. Keep frontmatter/body/log concise. Default raw-note frontmatter keys are `title`, `source`, `created`, `updated`, `type`, `domain`, `tags`, `topic_hints`, `github_links`, `huggingface_model_links`, `huggingface_dataset_links`, `capture_route`, and `captured`. The link fields carry direct source-exposed paper-owned GitHub/HF model/dataset artifact URLs. `tags` and `topic_hints` are phrase-level semantic routing cues. The body remains scientific reading prose; route, resource, probe, API, and audit details live in evidence, closeout, status, or log surfaces.
7. Run canonical closeout through the generated `closeout_command.preview.sh` / `closeout_args.json` after assembly; it pre-verifies, control-scans, marks pending, cleans temp, appends compact log, and reports status. Native prepare/refresh remains the separate native workflow when policy requires it. Open `references/raw-fast-closeout-ledger-contract.md` for repair or missing generated commands.
8. Run standalone verifier/status/validation only when repairing a failure or when compiled/meta/index surfaces changed.
9. If threshold/pre-query policy requires integration, clear wiki integration first, then run the native graph refresh required by `pending_native_refresh.json` as the immediate next step: generated closeouts only defer it via `--defer-native-refresh`, they do not cancel it. Native status exposes `next_refresh_kind`: ordinary cycles are `incremental`, and after 5 completed incremental graph updates the next run is `full-rebuild`. Every native graph update/rebuild must use vector cache via `--fill-missing-vectors`; closeout status-only output is not the refresh.

## Structured note and resource labels

Formulas/objectives go inside `## Methodology` using Obsidian-renderable `$...$` / `$$...$$` math delimiters. Figure/table evidence becomes prose conclusions beside the supported claim. Scan for non-whitespace control characters and non-rendering escaped MathJax delimiters before closeout. Treat verifier success as a structural gate: before marking an item done, reread the raw note against the structured contract and repair shallow prose rather than adding keywords to satisfy regexes.

Use conservative resource labels in closeout/evidence/ledger/log surfaces: `verified_present`, `verified_absent` after user-requested authoritative audit, `probe_failed`, `not_checked`, and `dependency_not_artifact`. Keep paper-owned artifact links scoped to direct source-exposed routes; put dependencies, mirrors, base models, third-party reproductions, broad family repos, and search/title hits in external evidence when relevant.

## Bounded raw-note-only manifest/status batches

Use this when the user gives manifest line ranges, a small batch list, or a worker/status-file protocol and wants structured raw notes now rather than compiled/meta/log integration. Preserve manifest line number, raw URL, source kind, intended target path, title/source ID, existing-note flag, temp path, caveats, and completion state in a compact status file under `${LLM_WIKI_RAW_FAST_BATCH_TMP_ROOT}/` or the configured task scratch root. Do not invent compiled/meta/log changes when the task is raw-note-only.

Search existing raw notes by exact IDs and title fragments before writing. Roundup/digest mentions are not canonical duplicates; reuse an existing canonical note only after confirming same paper/version/route identity. For OpenReview, use `source_page`, `source_pdf`, and `openreview_id`; do not invent arXiv fields. For arXiv-like routes, fetch abs/API/PDF/HTML/DOI/source when available, but API stalls are non-blocking if other evidence suffices.

Extract targeted method/results/limitations/resource windows from the generated source guidance before broader source or PDF reading. For TeX handoffs, read the generated filtered TeX source first; source-read plan entries are fallback locators for named gaps after that read. For non-TeX routes, source-read plan entries are targeted windows as the handoff indicates. Apply direct-link-only resource policy; keep provenance, direct-link results, and release-boundary caveats in external evidence, closeout, or compact status files. Verify every note plus status-file format, required source keys/URLs, headings, direct-link summaries, strict secret scan, and tmp cleanup. JSONL is valid one-object-per-line; whole-file `json.loads` will report `Extra data`. If a downstream consumer expects a JSON array, write one array and verify entry count.

Resource reminders: project page 200 is not a model/checkpoint/data release; GitHub/HF failures stay unresolved without fallback search; generic arXivLabs/CatalyzeX/HF widgets are not official resources; demo-only pages and companion repos are not full reproduction code; repo README-current claims must be separated from paper claims.

## Checklist-driven paper manifests

Use this when the user provides a local Markdown checklist/manifest and explicitly wants every paper clipped with each source line commented as soon as its note is complete.

1. Orient with `SCHEMA.md`, `index.md`, recent `log.md`, and duplicate searches across raw notes/maps by arXiv/OpenReview ID, exact title, acronym, and project/resource URL. Reuse existing canonical structured notes after same-paper identity is verified.
2. Parse stable work items: line number, raw URL, surrounding title/comment text, current DONE status, source kind, and intended canonical note path. Ignore links already inside `<!-- DONE ... -->` unless revalidation is requested.
3. For each missing canonical paper, follow structured-paper raw-fast; keep resource/probe details external.
4. Patch the checklist immediately after each item with a stable comment such as `<!-- DONE YYYY-MM-DD HH:MM: clipped to raw/clip/YYMM/<file>.md -->` or `<!-- DONE ... reused existing canonical note: raw/clip/YYMM/<file>.md -->`. Patch only the relevant line/region and verify no unrelated line changed.
5. Refresh `_meta` maps, `index.md`, and `log.md` only when the user requested a full integration/discoverability pass or batch policy requires it; raw-note-only batches may leave compiled totals unchanged.
6. Verify DONE coverage, new raw and reused canonical counts, duplicate guards, scoped wikilinks/source refs, image hygiene, secret scan, touched map/index/log parity, and temp cleanup. Report checklist path, DONE coverage, new/reused counts, files updated, verifier highlights, and final timestamp.

Pitfalls: do not delay DONE comments to the end; do not count roundup/newsletter mentions as canonical notes; do not duplicate existing canonical raw notes; do not over-integrate when the user asked only for raw notes; do not trust global `filename in file_text` checks for map updates; separate expected final-log hits from raw-note duplicates.

## Post-context-compression reconciliation

Use this when a preserved handoff resumes a single-paper ingest and write/refresh/verify may already be partly done. The latest visible user message wins; if it names a different source, discard stale state and route normally.

Before restarting fetch/dedup/write work, reconcile durable state: canonical raw note existence/sections, exact single `log.md` action block, `_meta/raw-clip-map.md` and `_meta/topic-map.md` source/body-row cues, active raw count and affected chronological-batch count, target compiled-page source refs/content cues, and temp dirs before cleanup.

If raw/compiled/meta edits landed but log is missing, clean temp dirs from a safe cwd, run scoped verifier before log append, append exactly one log block, then run a post-log smoke that separates hits as canonical raw, other raw, compiled/meta, log, or other. If a log exists but final smoke is missing, patch the existing block in place rather than appending a second ingest block.

For raw-fast captures, distinguish terminal states: threshold/pre-query integration ran, so verify wiki integration, validation, graph-refresh status, and ledgers; or below threshold, so raw note is saved/verified, pending wiki integration is recorded, native refresh is blocked by upstream wiki integration, and `_meta`/compiled/log work is intentionally deferred. Do not force compiled/meta/log/native work because preserved todos look unfinished. If the packaged scoped verifier is unavailable, reproduce only essential checks in one compact script: raw frontmatter/headings, raw-only duplicates, source refs, map counts, changed compiled/meta cues, image targets, wikilinks, index parity, temp absence, and strict secrets.

## Parallel staging and single-owner batch closeout

Use `ops.raw_fast_batch` when several papers should be read concurrently. Parallelism ends at staging: workers never write canonical notes, ledgers, `log.md`, integration plans, or native state.

Create one manifest and isolated workdirs from the wiki-graph workdir:

```bash
python3 -m ops.raw_fast_batch init --root "$LLM_WIKI_ROOT" --state-dir "$LLM_WIKI_STATE_DIR" --tmp-root "$LLM_WIKI_RAW_FAST_BATCH_TMP_ROOT" --url <paper-1> --url <paper-2>
```

The JSON result contains `manifest_path`, input-ordered items, unique `<input_index>-<source_hash>` workdirs, and one deterministic `prepare_command` per item. The parent runs those commands, then materializes concise source-aware worker contracts:

```bash
python3 -m ops.raw_fast_batch worker-contracts --manifest <batch_manifest.json>
```

Pass the returned `worker_tasks` unchanged to up to 10 leaf workers per wave. Each task points to a generated `worker_prompt.md` and compact `worker_contract.json`; identity comes from prepared `candidate_frontmatter.json`, not parent-authored prose. A worker owns the same complete quality loop as a single-paper clipping agent: it reads its generated handoff and required source surfaces, writes `raw_body_draft.md`, rereads the complete draft against the structured contract, may repair only `domain`, `tags`, or `topic_hints`, and runs the contract's `finish-worker` command. If programmatic validation fails, that worker repairs the named issue and reruns the finish command; only the script writes a `ready` `worker_result.json` with the exact prepared source identity. Workers batch independent reads/vision calls, use fallback spans only for a named gap, and make at most one targeted visual retry after the initial visual batch.

A child must not hand-write `worker_result.json`, run `raw_fast_note_assemble`, `raw_fast_closeout`, `batch_wiki_integration`, native refresh, or batch closeout; it must not edit `batch_manifest.json` or canonical wiki state. Pitfall: a batch item's generated `agent_handoff.md` keeps the single-paper "Next actions" text telling the agent to run `assemble_command.preview.sh` / `closeout_command.preview.sh`; for batch workers the `worker_prompt.md`/`worker_contract.json` win and those previews must NOT be run — the staged draft is published later by the parent's `ops.raw_fast_batch closeout`. Related single-paper pitfall: re-running assemble without `--overwrite-existing` never refreshes the existing note; the allocator takes the next free `YYMMDDNN` sequence and silently publishes a duplicate while rewriting closeout artifacts to the new path. If a batch worker already ran assemble, recover by removing every published variant of the note, re-running the packaged `sync_closeout_artifacts_after_publish(workdir, root=..., state_dir=..., raw_file=<contract provisional path>)`, then running the contract `finish-worker`; canonical log/_meta/manifest stay untouched because only assemble-side artifacts were written. The parent waits for every child to finish or explicitly fail and inspects only the script-owned manifest, contract/result status, source identity, and generated diagnostics needed for fan-in. The parent does not open paper sources, staged draft bodies, or canonical note bodies for semantic review, spot-checking, or content repair. Any item-level validation or quality failure goes back to the responsible worker for repair or explicit failure. After all results are terminal, the parent invokes closeout exactly once:

```bash
python3 -m ops.raw_fast_batch closeout --manifest <batch_manifest.json>
```

Closeout validates all staged notes outside the lock, then one parent writer acquires `wiki_mutation.lock`, recovers/reuses same-source canonical notes, allocates final `YYMMDDNN` paths in input order, creates new notes with exclusive create, batch-marks pending once, appends ordinary per-note log entries once, and makes one threshold decision. A failed worker does not block valid items. `--no-auto-integrate` records the decision without running integration. Batch automatic integration supports only the local runner; `--runner external` publishes raw/pending/log successfully but reports `external_runner_deferred` and never launches Hermes or an external command. Local integration always defers native refresh, so a successful integrated batch still reports `graph_status=pending` until the existing native pickup owner completes.

Treat report fields independently: `raw_status` describes canonical publication, `wiki_status` describes pending/integrated/review/failure, and `graph_status` describes native freshness. `ok=true` means closeout/report completed; it does not mean every worker succeeded or the graph is fresh. Counts are program-derived and `items` remain in input order.

Recovery reuses manifest-recorded paths only when the canonical content hash still matches, reuses an exact same-source canonical note, and idempotently avoids duplicate pending/log entries. A missing recorded file, hash/source conflict, lock timeout, invalid staged note, exhausted sequence, or exclusive-create conflict stops closeout while preserving workdirs, already published notes, and pending state. Rerun the same manifest after repairing the named blocker; no separate WAL or state mirror exists.

For ordinary single-paper clipping, keep using generated assemble/closeout previews. Single and batch flows share `ops.raw_fast_publish`; do not bypass it with direct `Path.write_text` for new canonical raw notes.

## What this file must not contain

Do not duplicate exact CLI blocks here; canonical commands live in `references/raw-fast-closeout-ledger-contract.md`. Do not store transient source identifiers, queue snapshots, scratch execution handles, or trial-local performance numbers in this default reference.
