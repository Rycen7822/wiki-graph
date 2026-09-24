# External verified raw-note intake and explicit integration

Use this when the user supplies a raw note already produced and verified by another/cloud agent and asks to add it to the local llm-wiki. Treat this as receiving a completed raw-fast raw note, not as a request to immediately integrate compiled/wiki surfaces or refresh native graph state unless the user explicitly asks for that later step.

## Default trigger

Use the default intake path when:

- the user attaches or points to a Markdown raw note outside the wiki root;
- the user says another agent already clipped / wrote / verified the raw note;
- the user asks to put it into llm-wiki but does not explicitly request immediate compiled-page integration, `_meta` synthesis, or graph-fresh retrieval.

The phrase “整合进 llm-wiki” does not override this rule when the user also says the raw note is already completed/verified elsewhere. In this class, “整合” means raw-note intake + pending queue unless immediate compiled/graph integration is explicitly requested.

## Default contract: queue intake only

Do only raw-note intake and waiting-queue increment:

1. Copy or place one canonical raw note under `raw/clip/<YYMM>/...md`.
2. Repeat local structural checks after the note is inside the wiki root.
3. Mark the note pending in `pending_wiki_integration.json` so the waiting queue count increases.
4. Append the compact raw-fast closeout log entry if using the closeout wrapper.
5. Stop and report the saved raw path plus pending queue status.

Do **not** do any of the following by default: run `python3 -m ops.batch_wiki_integration auto-integrate`, pass `--auto-integrate` to closeout or mark-pending, run `python3 -m ops.batch_wiki_integration clear-success`, update `_meta/raw-clip-map.md`, `_meta/topic-map.md`, `index.md`, or compiled pages, run native status/refresh, or claim graph freshness for the new note.

## Default intake workflow

1. Read the supplied raw note and identify title, source id / URL, created/captured time or filename prefix, domain, tags, and topic hints.
2. Search existing `raw/clip/**/*.md` for duplicate evidence using frontmatter-qualified source/title patterns when possible. If a canonical raw note already exists, do not create a duplicate; report the existing path or refresh it only if the user asks.
3. Choose the canonical target path from the note's clipping-time filename prefix when present. A valid current prefix is `YYMMDDNN_`: `YYMMDD` from clipping/raw-note creation date and `NN` as the chronological same-day sequence. The month directory is `raw/clip/<YYMM>/`. If the external filename is invalid, missing, or the user asks to treat it as current intake, assign the current intake date/sequence and mirror `created`, `updated`, and `captured` to the current intake time.
4. Preserve the external agent's raw-note content. Patch only local contract violations: forbidden frontmatter fields, standalone process/resource sections, embedded Markdown images, copied tables/figures, control characters, or broken YAML. If closeout fails because Methodology uses inline-code pseudo-formulas, convert the central mechanism/evaluation formula into real TeX math while preserving meaning. If control scan reports ASCII codepoint 12, check for accidental form-feed from unescaped `\frac`-style text and replace it with a literal TeX backslash sequence.
5. Run `python3 -m ops.raw_fast_closeout` using the canonical shape in `references/raw-fast-closeout-ledger-contract.md`, but omit `--auto-integrate`. Include `--fast-final-verify`, `--append-log`, the required `--raw-file`, `--title`, `--source-id`, literal `--pattern` values such as frontmatter-qualified source/title, compact `--topic-hint`, and `--resource-status-summary 'External raw note supplied as already verified; local closeout repeated raw-note structural checks only.'`.
6. If the closeout wrapper is unavailable or fails before ledger marking for a wrapper-specific reason, debug with `${WIKI_GRAPH_REPO}/.agents/skills/llm-wiki/scripts/raw_fast_note_verify.py`, then use `python3 -m ops.batch_wiki_integration mark-pending` directly without `--auto-integrate`. Keep the same title/source/topic/resource summary values.
7. Run `python3 -m ops.batch_wiki_integration status` only to confirm queue state if closeout output is insufficient. Desired final state: `pending_count` increased by one for the accepted note, with the new raw path present in pending/actionable items.
8. Do not run full `python3 -m ops.validate_wiki` solely for intake unless the raw note required repair or you changed wiki surfaces beyond raw note + compact log. Do not run native status/refresh.

## Explicit-only compiled/wiki integration

Continue past queue intake only if one of these is true:

- the user explicitly says to immediately update compiled/wiki pages, `_meta` maps, `index.md`, or wikigraph/native state for the supplied external raw note;
- a later query requires pending raw notes to be integrated before answering, and the user accepts that wiki integration should run now;
- the queue reached the normal integration threshold and the current task is a batch integration closeout, not a single-note raw intake.

Preconditions before explicit integration: the raw note exists under the canonical wiki root, local structural verification passed, the note is present in `pending_wiki_integration.json`, and the user/threshold/pre-query policy actually requires integration now.

Explicit integration workflow:

1. Orient with `SCHEMA.md`, `index.md`, recent `log.md`, `_meta/raw-clip-map.md`, `_meta/topic-map.md`, current `pending_wiki_integration.json`, and relevant raw/compiled pages.
2. Search existing raw/compiled pages for source id, title, and distinctive phrases. If this is a duplicate or canonical refresh, reconcile rather than creating another compiled anchor.
3. Read the accepted raw note and choose the smallest durable integration surface. Usually update existing `_meta` entries and existing compiled anchors; do not create one compiled page per note unless the theme warrants it.
4. Update `_meta/raw-clip-map.md`, `_meta/topic-map.md`, relevant compiled pages, and `index.md` timestamp only when compiled/wiki navigation changed. Append or patch a compact `log.md` entry describing placement and validation.
5. Run full wiki validation before clearing pending wiki integration.
6. Clear only the integrated note with `python3 -m ops.batch_wiki_integration clear-success --integrated-path <raw-path> --reason manual`, or let the threshold integration runner clear the batch. `clear-success` is correct only after wiki/meta/compiled integration actually happened and validation passed.
7. Check `python3 -m ops.batch_native_refresh status`. Run refresh only when status says it is unblocked and due for the current reason, or when the user explicitly asked for graph freshness.
8. After a real refresh, run retrieval smokes for the target raw clip and an expected raw-section. Patch `log.md` with final validation, ledger, refresh, and smoke results when those surfaces changed.

## Reporting

For default intake, report canonical raw path saved, local verification/final report path, pending wiki-integration queue count or waiting-queue increment, and the explicit boundary that compiled/wiki integration and native refresh were intentionally not run.

For explicit integration, report raw saved, wiki integration completed or still pending, native refresh pending/blocked/fresh, validation path, and retrieval smoke result if a real graph refresh ran. Never say the note has been integrated into compiled wiki pages or graph retrieval unless the integration/refresh steps above actually completed and were verified.

## Pitfalls

- Another agent's verification reduces source-reading work; it does not remove local structural verification after copying into the local wiki root.
- `--auto-integrate` is wrong for default external intake even if the queue threshold is reached; default intake wants queue growth, not immediate batch integration.
- Native refresh is never evidence that Markdown/wiki integration was done; keep raw saved, wiki pending, graph pending, and graph fresh states separate.
- Do not clear pending wiki integration, update compiled pages, or run native refresh merely because the supplied raw note was “verified elsewhere”.
