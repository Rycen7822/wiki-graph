# Raw-fast closeout and ledger contract

Use after a raw-fast raw note has been drafted **only when generated closeout artifacts are missing, invalid, or failed**. The happy path is script-first: `ops.raw_fast_ingest_prepare` writes `closeout_args.json` and `closeout_command.preview.sh`; prefer those generated artifacts and load this reference only for repair, manual fallback, timeout reconciliation, or existing-canonical exceptions. This file owns verifier arguments, closeout wrapper behavior, pending wiki integration, existing-canonical refresh handling, native graph blockers, timeout reconciliation, and failure policy.

## Inputs and invariants

- Roots: production wiki `${LLM_WIKI_ROOT}`; native workdir `${WIKI_GRAPH_REPO}`; production state `${LLM_WIKI_STATE_DIR}`.
- Raw file path is wiki-root relative, e.g. `raw/clip/<YYMM>/<file>.md`.
- Provide exact title, stable source id, literal duplicate patterns that occur in the note, phrase-level topic hints, and optional compact resource summary for ledger/log only. Keep ledger `source_id` and probe/resource summaries out of raw-note frontmatter.
- Temp path must be outside wiki root, native workdir, and system `/tmp`; canonical per-task root is `${LLM_WIKI_RAW_FAST_TMP_ROOT}/<slug>`. Run closeout from `${HOME}` or `/`, keep wrapper `--workdir` at the repo workdir, and finish only after temp absence is verified.

## Verifier command shape

Standalone verifier runs are for debugging or repair; routine closeout already pre-verifies before ledger mutation.

```bash
VERIFY="${LLM_WIKI_RAW_FAST_VERIFIER}"
python3 "$VERIFY" \
  --wiki "$LLM_WIKI_ROOT" \
  --raw-file raw/clip/<YYMM>/<file>.md \
  --structured-paper \
  --patterns '<stable-source-id>' '<exact-title>'
```

Rules: use `--raw-file`, not `--raw-path`; patterns are literal substrings and every supplied pattern must hit; prefer separate stable literals such as versioned arXiv id, canonical source URL, and exact/frontmatter title. Do not use unsupported `--arxiv-id`, `--source-id`, or `--title` verifier flags. Section enforcement belongs to `python3 -m ops.batch_wiki_integration mark-pending --required-section ...` or wrapper defaults.

## Closeout wrapper command shape

Use `--auto-integrate` only when threshold/pre-query policy should be honored. Current `ops.batch_wiki_integration` defaults auto-integration to deterministic local `integrate-local` / `apply-plan` execution for machine-owned plans; Hermes/external integration runners are opt-in through explicit runner/command flags. Generated closeout previews carry `--defer-native-refresh`: wiki integration (apply/validate/clear) stays inline in the closeout and returns in about 60-90s, while the native graph refresh stays pending in `pending_native_refresh.json` and belongs to a separate follow-up run of `python3 -m ops.batch_wiki_integration refresh-native-after-integration --root <root> --state-dir <state-dir> --workdir <workdir>` (run it in `terminal(background=true, notify_on_complete=true)`; it owns semantic rebuild, workspace build, cutover, restart, and the configured local embedding Docker start/health/stop lifecycle). Direct `ops.batch_wiki_integration` calls without `--defer-native-refresh` keep the old inline behavior: treat them as potentially long bounded jobs whose outer timeout must exceed the integration/native-refresh budget. Do not interpret a foreground timeout as closeout success or failure without durable-state reconciliation. For externally supplied already-verified notes, load `references/external-verified-raw-note-queue-intake.md` and omit it.

```bash
VERIFY="${LLM_WIKI_RAW_FAST_VERIFIER}"
cd "$WIKI_GRAPH_REPO"
python3 -m ops.raw_fast_closeout \
  --root "$LLM_WIKI_ROOT" \
  --state-dir "$LLM_WIKI_STATE_DIR" \
  --workdir "$WIKI_GRAPH_REPO" \
  --output-mode compact \
  --verifier "$VERIFY" \
  --raw-file raw/clip/<YYMM>/<file>.md \
  --title '<exact title>' \
  --source-id '<stable source id>' \
  --pattern '<literal stable id>' \
  --pattern '<literal exact title>' \
  --topic-hint '<topic>' \
  --resource-status-summary '<compact summary>' \
  --tmp "${LLM_WIKI_RAW_FAST_TMP_ROOT}/<slug>" \
  --allow-non-tmp-cleanup \
  --fast-final-verify \
  --append-log \
  --auto-integrate \
  --defer-native-refresh \
  --native-refresh-mode status
```

`ops.raw_fast_closeout` refuses cleanup outside `/tmp/` unless `--allow-non-tmp-cleanup` is present. Use that flag only after confirming `--tmp` is the intended per-task root, not wiki root, native workdir, home directory, or sensitive home subdirectory. Required current args are `--raw-file`, `--title`, and `--source-id`; do not use legacy `--raw-note`, `--tmp-root`, `--queue-wiki`, `--cleanup-tmp`, or `--print-json`.

Generated previews and the manual fallback above use `--output-mode compact`: successful closeout prints the short session summary, while failures print only the failing stage and relevant diagnostics. Direct CLI calls remain `full` by default for machine-readable compatibility.

`--native-refresh-mode status` is the normal closeout mode: it records native status without running `batch_native_refresh refresh --prepare-only`. With `--defer-native-refresh` this status keeps reporting `should_refresh=true` until the separate refresh owner path above runs and cuts over; that is expected, not a failure. Pass `--native-refresh-mode prepare` only when an operator explicitly wants a prepared native workspace; prepare-only output is not live graph freshness and still requires the native operations router for cutover/restart/retrieval smoke.

## Pre-closeout hygiene

- Clear stale loose workdir artifacts from prior runs before bundle/probe tools: PDFs, extracted text, `docling.*`, e-print/source dirs, localized figure renders, inventories, probes, and skeletons.
- Use one slug/tmp variable. Verify all compact evidence lives under the exact `--tmp` path and remove wrong-slug temp roots from a safe cwd.
- Treat generated frontmatter/skeleton/next-path/evidence artifacts as internal or advisory according to the generated handoff; use authoritative metadata for title, versioned source id, source route, and filename date.
- Confirm versioned arXiv/API/abs/HTML/PDF/e-print/source identity; resolve latest version for unversioned inputs.
- Preserve `<tmp>/evidence_bundle.json`, `agent_handoff.json/md`, `closeout_args.json`, `closeout_command.preview.sh`, `agent_brief.json/md`, `evidence_report.json/md`, `note_candidate.json/md`, and any compact failure-triggered manual resource/visual/probe reports before cleanup; full health/probe detail belongs under `${LLM_WIKI_STATE_DIR}/raw_fast_reports/*`. Build `evidence_bundle.json` with top-level `source_url`, `title_guess`, and `kind` plus `preflight`/`agent_automation` when available so closeout's evidence summary remains self-describing after temp cleanup.
- Prefer `<tmp>/closeout_args.json` and `<tmp>/closeout_command.preview.sh` produced by `ops.raw_fast_ingest_prepare`. If those files are missing or invalid, derive closeout literals with `ops.raw_fast_closeout.derive_closeout_args_from_bundle(<tmp>/evidence_bundle.json)` instead of hand-assembling `--pattern`, `--topic-hint`, and `--resource-status-summary`. After closeout, use `build_raw_fast_session_summary(final_report)` or the final report's compact fields instead of rereading log/status surfaces unless repairing a failure.

## Required sequence

1. Let closeout pre-verify before ledger mutation; run standalone verifier first only for debugging.
2. Scan formula-heavy notes for non-whitespace ASCII control bytes and broken math delimiters; restore intended TeX commands.
3. Mark pending only through `python3 -m ops.batch_wiki_integration mark-pending`; never edit ledger JSON by hand.
4. If `--auto-integrate` and policy require integration, run wiki integration first; after successful `clear-success`, immediately run the native graph refresh indicated by native status. Use `--fill-missing-vectors` for every graph update/rebuild so `vector_cache.sqlite` is used/filled. Ordinary cycles are `incremental`; after 5 completed incremental graph updates, `next_refresh_kind` becomes `full-rebuild`.
5. Clean declared temp paths from a safe cwd.
6. Prefer `--fast-final-verify`; run a second full verifier only if note/wiki surfaces changed after pre-verify.
7. Use closeout's wiki/native gate fields. If `blocked_by_pending_wiki_integration=true`, do not claim graph-ready pending is zero; run standalone native status for exact counts.
8. Keep `log.md` compact via `--append-log`; detailed evidence stays in evidence bundles or native state reports. Patch only newly appended compact bullets when critical status or blank-line separation is wrong.
9. Do not run `python3 -m ops.validate_wiki` solely after raw-note save or compact log append. Use full validation for `_meta`/compiled/index rewrites, batch integration, query consolidation, or troubleshooting.

## Ledger and status commands

Names: `python3 -m ops.batch_wiki_integration mark-pending` uses `--raw-path`; verifier/closeout use `--raw-file`; `auto-integrate --dry-run` is the dry-run form; native refresh pending state is `pending_native_refresh.json` under the explicit state dir.

Status commands put subcommand before options:

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

Do not put `--root` / `--state-dir` before the subcommand. For inline JSON summaries, avoid here-docs that consume Python stdin; use `python3 -c` or `subprocess.check_output(...)`.

## Existing-canonical refresh exception

Do not use normal closeout if refreshing an already canonical raw note would append a duplicate pending wiki item.

- If no durable note text changed, save useful external evidence and report no-write confirmation; do not mutate wiki/native ledgers.
- If the same raw path is already in `pending_wiki_integration`, update that identical `raw_path` entry and verify actionable count stayed unchanged.
- If the note is wiki-integrated and only raw-note facts changed, do not re-add pending wiki integration; mark native refresh pending with reason `raw-note-contract-refresh`, then read wiki/native statuses and report graph refresh blocked if any upstream wiki integration is pending.
- For refreshes bypassing normal closeout, write `<tmp>/evidence_bundle.json` and durable `${LLM_WIKI_STATE_DIR}/raw_fast_reports/<raw-name>_refresh_resource_report.json` before cleanup.
- When needed, verify with `${WIKI_GRAPH_REPO}/.agents/skills/llm-wiki/scripts/raw_fast_note_verify.py --structured-paper --tmp <dir> --patterns <canonical-id/title>`; after deleting temp root from safe cwd require `raw_fast_ok=true`, no extra frontmatter, no Markdown image embeds, `strict_secret_hits=0`, and `tmp_absent=true`.
- Append/update log only for durable note/report/ledger changes; run `python3 -m ops.validate_wiki --write-report` only if log/meta/index/compiled surfaces changed.

## Native graph boundary

Do not run native refresh/cutover over raw-fast pending notes. Read native status together with wiki-integration status; native refresh is considered only after wiki integration validates and clears. After wiki integration clears, native graph refresh is required by policy: use `next_refresh_kind` from native status, run with `--fill-missing-vectors`, treat `incremental` as the ordinary graph update, and run `full-rebuild` after 5 completed incremental graph updates. Prepare-only evidence is not live graph freshness. For live freshness, load `references/wikigraph-native-operations-router.md` and run explicit cutover/restart, `/health`, `/query/data`, storage watch, pending clear, service replacement, and rollback checks.

## Timeout and reconciliation policy

If closeout, auto-integration, or native preparation times out, do not rerun blindly. Reconcile durable state first: final raw-fast report, `pending_wiki_integration.json`, `pending_native_refresh.json`, latest wiki integration run, latest native refresh reports, tracked process state, and service `/health` when relevant. If the outer Hermes terminal timed out while `--auto-integrate` was waiting on local integration/native refresh (or an explicitly configured external runner), expect the closeout wrapper's post-mark cleanup/final report to be missing until repaired; verify raw note state, remove only the declared per-task tmp from a safe cwd, then rerun the standalone verifier with `--tmp` or rerun closeout only after confirming no duplicate pending mark will be added. Treat runner output as self-report until verified: confirm `pending_wiki_integration=0`, `last_successful_integration_raw_count` equals active raw count, and native pending was marked/cleared before reporting success or starting refresh. Threshold integration or live cutover closes with retrieval smokes, log patch when needed, and final validation.

## Failure policy

- Verifier/control-scan failure: do not mark pending; patch raw note and rerun.
- If pre-verify passes but closeout fails at `stage=control_scan`, suspect raw-note bytes rather than schema: formula-heavy note bodies written through Python normal strings can silently turn TeX commands into ASCII controls (`\tau` -> tab, `\bar` -> backspace, `\frac` -> formfeed, `\alpha` -> bell, `\v...` -> vertical tab) before the verifier notices. Reopen the canonical raw file, inspect the reported line/column with `repr`, rewrite the affected note from a raw string or external body draft, and require `control_count=0`; for TeX-heavy notes also eliminate/inspect literal tabs because they often mean a lost `\tau`.
- For formula-heavy raw-note generation, choose a brace-safe composition path from the start: keep the note body in a raw string or external body draft, compose frontmatter separately, or use non-brace sentinel placeholders outside TeX before interpolation. After the canonical write, rerun the standalone verifier plus closeout so TeX braces and backslashes are checked on the actual saved file.
- Pending mark failure: raw note remains saved, but ledger failure is explicit. Reconcile wiki/native ledgers before rerunning closeout because threshold integration may already have succeeded even when its follow-through native refresh returned nonzero.
- If native follow-through fails with `ModuleNotFoundError: zvec`, verify the exact interpreter used by generated closeout/restart commands. These normally use `/usr/bin/python3`; a project `.venv` alone does not repair that path. Install the declared project runtime dependencies into that interpreter's enabled user site with uv (for this runtime, `uv pip install --target "$(python3 -m site --user-site)" -e ${WIKI_GRAPH_REPO}`), verify `python3 -c 'import zvec, starlette, uvicorn'`, remove any generated untracked `*.egg-info/`, then rerun only the guarded native refresh after confirming wiki pending is already zero.
- Wiki integration failure: do not clear pending; report validation path and next action.
- Native refresh failure: keep Markdown/wiki status separate from graph freshness; reconcile durable state before calling a timeout failed.
