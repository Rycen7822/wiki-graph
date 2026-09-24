---
name: llm-wiki
description: "Thin router for llm-wiki clipping/query. Single bare paper links — arXiv, OpenReview forum/pdf?id/attachment, DOI, direct PDF, Hugging Face paper, DAIR academy paper — mean structured raw-fast clipping by default; explicit multi-paper batches use isolated staging and parent-only closeout."
version: 2.5.71
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [wiki, knowledge-base, research, notes, markdown, rag-alternative]
    category: research
    related_skills: [obsidian]
---

# llm-wiki

Canonical thin router for the user's layered Markdown llm-wiki. Keep this file short: it carries trigger rules, hard boundaries, invariants, and the default simplified-reference route. For commands, branch mechanics, cases, and troubleshooting, load only the simplified reference or the exact paths returned by automation failure output.

## Project-local path bootstrap

This tracked skill lives under the wiki-graph repository's `.agents/skills/llm-wiki/`. Local filesystem paths belong only in the repository's ignored `.env`, not in this skill or Git. Before running wiki operations from the repository root, load **path keys only** from that file in the current Bash session:

```bash
WIKI_PATH_EXPORTS="$(python3 .agents/skills/llm-wiki/scripts/load_env_paths.py --shell)" &&
eval "$WIKI_PATH_EXPORTS" &&
cd "$WIKI_GRAPH_REPO"
```

The trusted helper shell-quotes an explicit path-key allowlist and never emits API keys or other credentials. It fails closed when `.env` is missing, a required path key is absent/non-absolute, or `WIKI_GRAPH_REPO` does not identify this repository. Required keys are `WIKI_GRAPH_REPO`, `LLM_WIKI_ROOT`, `LLM_WIKI_STATE_DIR`, `LLM_WIKI_STORAGE_ROOT`, `LLM_WIKI_RAW_FAST_TMP_ROOT`, and `LLM_WIKI_RAW_FAST_BATCH_TMP_ROOT`; optional keys are `LLM_WIKI_OBCLIP_WRAPPER`, `LLM_WIKI_OBCLIP_PROFILE`, and `OPENREVIEW_ENV_PATH` (a credential-file location, never its contents). The project skill directory and verifier path are derived from `WIKI_GRAPH_REPO`, not stored as machine-specific literals. Run `python3 .agents/skills/llm-wiki/scripts/load_env_paths.py --check` for a value-free preflight. Never commit or display `.env` or generated path exports. Read-only inspection of this skill does not require loading `.env`.

## Hard boundaries

- Production wiki root: `${LLM_WIKI_ROOT}/`. Mutate it only for explicit wiki edit/ingest/integration tasks.
- Research idea drafts default to `${LLM_WIKI_ROOT}/idea/` with `type: idea`. Keep that directory out of `index.md`, `_meta` maps, wiki-integration queues, and native graph discovery by default; only promote a specific idea after explicit user direction.
- Native runtime/workdir: `${WIKI_GRAPH_REPO}/`.
- Central storage root: `${LLM_WIKI_STORAGE_ROOT}/`; production native state is `${LLM_WIKI_STATE_DIR}`.
- Per-task scratch root: `${LLM_WIKI_RAW_FAST_TMP_ROOT}/<slug>/`. Put source downloads, evidence, probes, extracted figures/tables, and drafts there; closeout must remove the whole per-task root.
- Never write native state/artifacts, generated virtual docs, service logs, sidecars, evidence trails, downloaded PDFs, source bundles, probe output, or scratch files into the human wiki root. Only canonical repo `ops.*` modules or packaged skill scripts may write durable ledgers/reports under the explicit native state dir.
- Raw notes preserve source-grounded reading notes and provenance; compiled pages hold durable interpretation; `_meta` pages organize navigation; native zvec state is auxiliary retrieval state, not source truth.

## When to use

Use this skill for llm-wiki ingest, query, lint, cleanup, organization, raw-note repair, structured paper clipping, blog/article clipping, query-page synthesis, native zvec+lexical status/retrieval/freshness, or any task touching `raw/`, compiled pages, `_meta`, `index.md`, `log.md`, the native runtime, or native state.

Bare links are actionable and should trigger this skill before generic web/PDF handling: a message containing only a paper/report link means structured paper raw-fast clipping by default. This includes arXiv URLs, OpenReview `forum?id=...`, `pdf?id=...`, and `attachment?id=...&name=pdf` URLs, DOI/Publisher pages, direct PDF links, Hugging Face paper pages, and DAIR academy paper pages (`academy.dair.ai/papers/<slug>-<arxiv-id>`). A bare ordinary blog/article/web/WeChat/X/Hugging Face Space link means ordinary webpage clipping by default: load `references/webpage-obclip-clipping.md` only when browser-backed capture, obclip, profile reuse, dynamic waits, image localization, or WeChat/X repair is needed; load `references/webpage-complete-translation-clipping.md` when the user asks for complete original clipping plus Chinese translation. For paper-like raw-fast clipping, first load `references/raw-fast-automation-quickstart.md`, then follow its `.env`-backed command with `--profile env` and explicit root/state/tmp paths; the script's default `prod` profile does not load root/state from `.env`. For arXiv clipping, do not load the `arxiv` skill first; it is support-only for explicit search/API/metadata fallback after the automation handoff asks for it. Do not load manual branch references unless that script or its generated handoff exits/stops with `manual_required=true`, `resource_review_required=true`, or exact `manual_reference_paths`. Ask only when the URL class or access is genuinely unclear. If another extraction/summary path has started and the user says “剪藏”, switch to llm-wiki ingest instead of delivering a standalone summary.

## Always-on invariants

1. For durable wiki edits, orient with `SCHEMA.md`, `index.md`, and recent `log.md` unless the loaded reference explicitly allows fast raw-only closeout.
2. Search existing raw and compiled pages before writing; avoid duplicates and refresh the existing canonical raw note when appropriate. Before any overwrite, record the canonical `created`; after assembly, re-read the written frontmatter and restore that value before closeout if the assembler reset it.
3. Reopen the exact source or canonical raw note immediately before final synthesis; never answer from hit lists alone.
4. Use `updated: YYYY-MM-DD HH:MM`.
5. For `raw/clip/<YYMM>/...md`, `<YYMM>` and the `YYMMDDNN` prefix come from clipping/raw-note creation date; `NN` is chronological same-day sequence. Do not use hour/minute, source date, publication date, arXiv month, or arXiv id.
6. Raw-note frontmatter is minimal but retrieval-friendly: default keys are `title`, `source`, `created`, `updated`, `type`, `domain`, `tags`, `topic_hints`, `github_links`, `huggingface_model_links`, `huggingface_dataset_links`, `capture_route`, and `captured`. Use `domain` for the broad semantic area (`machine-learning`, `alignment`, `robotics`, `computer-vision`, etc.), not for the source form; new notes should not use `domain: paper`. Paper-ness is carried by `type: raw-note`, `capture_route`, the source route, and the structured H2 contract. Keep `github_links`/`huggingface_model_links`/`huggingface_dataset_links` to direct source-exposed paper-owned URLs only; never put Hugging Face paper pages, HF search/index pages, HF collections, Spaces, project pages, link-health/probe status, API details, or broad-search results there. Keep `tags`/`topic_hints` specific, but do not put URLs, arXiv/source IDs, DOI/source identifiers, resource/probe/status facts, checksums, authors, categories, or route-specific metadata in them.
7. Raw-note body must not contain bare URLs/Markdown links, process headings such as `Evidence trail` or `资源与复现状态`, link-health/API/probe facts, project/GitHub/HF/PWC route details, repo/license/file-count/runtime audit detail, or “link reachable but not audited” caveats. Keep resource/probe/reproducibility details only in tmp/evidence, state reports, closeout reports, ledgers, or compact `log.md` entries.
8. Resource probes are direct-link-only by default. Do not search GitHub/HF/project pages by title, arXiv id, method name, author, or repo owner unless the source directly exposes a paper-owned route or the user explicitly asks for a reproducibility/resource audit. Classify source-exposed URLs by local context before treating them as paper-owned artifacts.
9. Structured paper notes integrate formulas/objectives into `## Methodology` using Obsidian-renderable `$...$` or `$$...$$` math; figure/table evidence becomes prose conclusions near the supported claim. The conclusion must include the key visual/table anchors that make it checkable: numeric values when present, row/column names, axis names and coordinates, metric changes, ablation directions, channel/question fields, or named examples/mappings for conceptual figures. Do not inventory chart layout, and do not stop at broad claims such as “supports the core conclusion” without the data/fields that support it. Do not create standalone formula/figure buckets, embed images, retain copied charts, or reproduce tables wholesale. Treat verifier pass as structural only; after prepare and before source-span reading/drafting, follow returned `writing_contract_refs`; before closeout, use `references/structured-paper-note-contract.md` sections `Raw-note quality bar before closeout` and `Minimum completeness floor` to check sample-derived section quality and perform semantic repair, not keyword stuffing.
10. When the generated handoff exposes TeX/e-print source refs (`tex_agent_ir_audit.md`, `paper_source.agent.tex`), read the audit and filtered original TeX as the first-pass source surface; treat `source_read_plan`/`source_tex` as fallback locators only after naming a concrete formula, table/figure conclusion, limitation, or uncertain-claim gap. Figure conclusions should follow handoff `visual_inspection` guidance and use `image_path` / `localized_path` anchors rather than caption-only summaries.
11. For long reports, dense appendices/source bundles, surveys, or user requests to preserve subtle details, load the context-safe long-source reference and use scratch notes/evidence batches before final raw-note write.
12. Already verified raw notes supplied by another/cloud agent are raw-fast intake only: repeat local structural verification, mark pending wiki integration, and stop unless the user explicitly asks for immediate integration.
13. Never run native refresh/cutover over raw-fast notes that are still pending wiki integration. Report raw saved / wiki pending / graph pending / graph fresh separately.
14. Avoid hard-wrapping natural-language prose in wiki logs, notes, or agent handoffs.
15. For an explicit multi-paper parallel batch, load `references/raw-fast-batch-wiki-integration.md`; after item prepares, run `ops.raw_fast_batch worker-contracts` and pass its `worker_tasks` unchanged. Each worker owns the same complete source-reading, drafting, full-note semantic self-review, and repair loop as single-paper clipping; it runs the generated finish command, repairs any validation failure internally, and reports `ready` only through the script-owned result. The parent waits for every success or explicit failure and inspects only generated contracts/results, item status, and source identity for fan-in—not source text, draft bodies, or canonical note content. Route an item failure back to that worker instead of repairing it in the coordinator. Only `ops.raw_fast_batch closeout` publishes canonical notes, mutates shared queues/logs, and makes the single integration decision.

## Default simplified-reference route

For paper-like raw-fast clipping, use this order: read this `SKILL.md`, read `references/raw-fast-automation-quickstart.md`, run the automation script named there, then follow the generated `agent_handoff.md` on success. For an explicit multi-paper batch, the quickstart routes to `references/raw-fast-batch-wiki-integration.md` and `ops.raw_fast_batch`; use generated worker tasks unchanged, keep workers staging-only, and let the parent own the single closeout. If the script exits nonzero or the handoff marks a manual stop, read only the exact `manual_reference_paths` returned by that output. For ordinary webpages, load `references/webpage-obclip-clipping.md` only when the route needs obclip/browser mechanics; load `references/webpage-complete-translation-clipping.md` only for complete original + Chinese translation tasks. The main router intentionally does not list fallback paper references; the automation output is the owner of that choice.


## Finish checks

- Loaded the smallest matching references and no obsolete backend path.
- Confirmed production root, workdir, central storage, and native state separation before writing.
- Verified source refs, wikilinks, frontmatter, raw-note body constraints, index/log/meta scope, and absence of wiki-root machine artifacts for the selected edit.
- Loaded `references/raw-fast-automation-quickstart.md` before running default paper-like raw-fast automation; loaded manual references only when script output/handoff explicitly returned exact `manual_reference_paths`.
- For a parallel batch, verified unique item workdirs, script-generated worker contracts/results, unchanged delegation tasks, staging-only children, complete parent fan-in, one canonical closeout, and separately reported raw/wiki/graph status.
- Reported raw saved / wiki pending / graph pending / graph fresh without implying native graph freshness when wiki integration or native pending queues block it.
