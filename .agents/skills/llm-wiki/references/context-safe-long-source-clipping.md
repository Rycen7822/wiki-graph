# Context-safe long-source clipping

Use this when clipping a long paper, vendor report, supplement bundle, source archive, benchmark dossier, dense appendix, or any source set large enough that a one-pass read/write could lose details or stall. This file owns the llm-wiki long-source scratch loop and temp-root boundary.

## Mandatory triggers

Load this when any of these hold:

- The user asks to read while taking temporary notes, draft in sections, reread the draft, avoid context-compression loss, preserve engineering experience, preserve subtle insight, or avoid forgetting details.
- The source is a long technical report, long paper, 50+ page PDF, long appendix/supplement, report plus code/source bundle, many-page HTML/PDF, dense table/figure source, or a 30–40 page paper with large TeX/source/appendix evidence.
- The task needs engineering detail: training-scale numbers, infrastructure design, serving substrate, fault tolerance, data filtering/order, ablation caveats, benchmark protocol, safety mitigations, throughput/goodput, or reproducibility boundaries.
- Extraction produces large text, many source files, many figures/tables, or multiple probe/evidence artifacts that cannot be reasoned over in one stable context window.

## Core rule

Do not read a large corpus and then write the final raw note from memory. Read small thematic batches, write scratch notes after each batch, draft from those notes, reread the complete draft, reopen sources for weak/exact claims, then copy to the canonical raw note only after confidence review.

## Scratch location and boundaries

Use one per-task root: `${LLM_WIKI_RAW_FAST_TMP_ROOT}/<slug>/`. Put scratch notes and draft raw notes under `<root>/scratch/`; put PDFs/source bundles/extracted text/rendered figures/probes/evidence sidecars under sibling directories inside the same root.

Never put scratch notes, extraction manifests, hashes, probe transcripts, draft ledgers, or source bundles under the human wiki root, compiled wiki directories, native workdir/state scratch, repo runtime workdirs, or system `/tmp`. Raw notes contain only the final reading note with compact frontmatter and integrated evidence. Detailed process evidence stays temporary until closeout removes the root after preserving necessary compact reports.

## Required workflow

1. **Scope note first.** Write `00_scope_requirements.md` with user request, source URL/path, target raw path or duplicate state, wiki root, scratch paths, source families, constraints, and confidence boundary.
2. **Size/source inventory.** Write `01_size_inventory.md` with downloaded files, extracted text paths, page/file counts, line/byte counts, page/section map, candidate figures/tables, and existing canonical raw-note status.
3. **Thematic reading notes.** Split by source structure: framing/abstract, method/architecture, training/data, systems/inference, experiments/tables, limitations/appendix, resources/probes. After each batch, write notes such as `02_method_notes.md`, `03_system_notes.md`, or `04_eval_notes.md` with page/file refs, exact numbers, formulas/tables/figures, implications, and unresolved questions.
4. **Incremental draft.** Write `06_draft_raw_note.md` section by section from scratch notes. Keep the structured raw-paper H2 contract; integrate formulas/figures in Methodology/Results/Limitations rather than standalone evidence buckets.
5. **Reopen source during drafting.** For exact names, parameters, equations, table values, figure axes, benchmark protocols, artifact status, or caveat wording, reopen the PDF/source/probe slice before writing or patching the claim.
6. **Full-draft reread.** After drafting, read the entire scratch draft. Check headings, stale claims, vague wording, source-boundary statements, duplicated sections, code fences, compact frontmatter, wikilinks, and whether engineering details/subtle insights survived.
7. **Confidence review.** If weak areas remain, reopen source slices, patch the draft, and record fixes in `07_confidence_review.md`.
8. **Copy only after review.** Copy the reviewed draft to the canonical raw note path only after full-draft reread passes. Preserve `created` when refreshing; update `updated` / `captured` using current wiki timestamp format. If the clipping crossed an hour/day boundary, patch scratch frontmatter and filename prefix before canonical write. Do not leave a second durable draft after closeout.
9. **Final raw-note verification.** Reread the final file head or relevant chunks and check required H2 sections, no deprecated process/resource headings, compact frontmatter, balanced fences, no non-whitespace control chars, expected high-value terms, and duplicate patterns suitable for closeout.
10. **Compact evidence before cleanup.** Before closeout, merge durable audit facts into `<tmp>/evidence_bundle.json`: source identity, page/source inventory, exact-link health for source-exposed routes, table/figure-derived values used in the note, and scratch/confidence note list.
11. **Close out through raw-fast tooling.** Run canonical closeout with safe duplicate patterns, `--tmp ${LLM_WIKI_RAW_FAST_TMP_ROOT}/<slug>`, `--fast-final-verify`, `--append-log`, and `--auto-integrate` when appropriate. The declared temp path must contain scratch plus source/evidence artifacts so final verification proves `tmp_absent=true`; report raw/wiki/graph freshness boundaries honestly.

## Subagent usage

Use focused subagents only to reduce context pressure. Give each child a narrow page/file range and require source-grounded notes with refs, numbers, mechanisms, caveats, and uncertainty. Split very long proof/source sections into smaller ranges such as one theorem block, 100–250 source lines, or one thematic subsection. Treat child output as notes, not verified final text; parent must cross-check source-critical claims, link health, resource status, duplicate absence, files written, and probe success against actual files/reports/status before finalizing.

## Quality gates

- Scratch notes exist for scope, inventory, thematic evidence, draft, and confidence review under `${LLM_WIKI_RAW_FAST_TMP_ROOT}/<slug>/scratch/` while clipping is active.
- Final raw note preserves mechanisms, non-obvious constraints, numerical settings, evaluation protocol boundaries, author caveats, and subtle insights, not just headline scores.
- Dense short reports still use the scratch loop when technical detail / engineering experience / insights must be preserved.
- Benchmark-table, accuracy/latency, ratio, average, or delta claims are tool-computed from source rows before writing; say “competitive but not uniformly best” when that is what tables show.
- Survey/review notes preserve taxonomy, method families, construction axes, sub-method routes, representative works, trade-offs, failures, benchmark/application coverage, and roadmap discussion.
- Theory/analysis notes preserve assumptions, definitions, key equivalences, objective-to-gradient transformations, theorem/proof skeletons, induced-boundary or limiting-distribution arguments, and caveats.
- Formula/table/figure/protocol claims sit beside the supported Methodology/Results/Limitation claim.
- Final closeout reports `raw_fast_ok=true`, or the blocker is reported without claiming wiki/graph freshness.

## Pitfalls

- Do not use wiki roots, native state/workdir, or system `/tmp` as scratch. This workflow's only allowed per-task temp root is `${LLM_WIKI_RAW_FAST_TMP_ROOT}/<slug>/`.
- If legacy scratch/process files are found under native `state/scratch` or old raw-fast roots outside the configured scratch directory, clean them from a safe cwd only after confirming they are not active inputs.
- Do not let extraction manifests, page digests, helper skeletons, or a single large model context replace thematic scratch notes and confidence review.
- Do not accept subagent summaries without source spot-checks.
- Do not copy an in-progress draft to the canonical raw note before full-draft reread and confidence review.
- Do not preserve full scratch as durable evidence; keep only compact closeout reports when needed, then remove the whole per-task temp root.
