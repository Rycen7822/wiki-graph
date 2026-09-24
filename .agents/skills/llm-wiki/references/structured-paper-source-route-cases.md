# Structured paper source and route cases

Use after `references/structured-paper-ingest-router.md` selects a paper route needing source identity, PWC/HF parsing, first-party artifact classification, project-page handling, or audit-mode expansion. Default exact-link triage lives in `ops.raw_fast_ingest_prepare`; manual helper commands live behind script-returned `manual_reference_paths` or explicit user audit scope. Note structure lives in `references/structured-paper-note-contract.md`; closeout lives in `references/raw-fast-closeout-ledger-contract.md`.

## Non-negotiable boundary

Default clipping verifies route reachability only. It is not resource discovery and not a reproducibility audit. If no direct paper-owned artifact route is exposed, record `not_checked` externally and do not search by title, arXiv id, DOI, method, author, lab, project-page scripts, dependency names, or repo guesses. If a direct route exists, run exact-link health only; do not inspect repo trees, licenses, releases, HF siblings/card/checkpoints/dataset rows, package files, or Spaces source unless the user explicitly requests an audit. Raw notes remain source-clean: route URLs, link-health, repository/license/runtime/audit facts, and skipped-audit wording stay in temp evidence, reports, closeout, ledgers, compact `log.md`, or final response.

## Route identity before artifact classification

- Resolve canonical paper identity first: title, source id, version, supplied route, and canonical route discovered through arXiv/API/DOI/PWC/HF metadata.
- Treat Hugging Face, PapersWithCode, alphaXiv, and DAIR academy paper pages as discovery/provenance routes, not proof that platform-owned artifacts exist. For numeric paths that expose an arXiv id, trust `ops.raw_fast_ingest_prepare` to route source reads through canonical arXiv metadata/e-print/PDF and keep the supplied route external.
- Resolve canonical source identity before artifact classification, then inspect paper-owned source links and HF/PWC APIs only when needed.
- Preserve route mismatches until title/source identity is verified. If a supplied route already has a canonical raw note, refresh only after exact duplicate search and route/version confirmation.

## TeX/e-print filtered source views

When arXiv/e-print evidence succeeds through TeX source, the generated handoff exposes `paper_source.agent.tex` plus `tex_agent_ir_audit.md`. These are script-owned reader surfaces in the evidence workdir, not raw-note content and not durable wiki artifacts.

Default reading order: read `paper_digest.md` as an evidence index, then `tex_agent_ir_audit.md` for coverage/flatten diagnostics, then `paper_source.agent.tex` as the filtered original-TeX source. The filtered source removes preamble/style/author formatting and references while preserving body source, equations, tables, figures, and appendices. Use `source_read_plan`/`source_tex` only as exact-span locators after naming a concrete formula, table/figure conclusion, limitation, or uncertain-claim gap.

Treat `latexpand` flattening as an audit signal, not the sole truth source; the script-owned include graph/source-unit map remains the route owner. `image_path` is a workdir-relative protected visual anchor and `localized_path` is a localized copy when available; if handoff asks for visual inspection, inspect the image/table surface instead of writing caption-only conclusions. If these paths are missing or unresolvable, stop and repair/rerun evidence generation rather than papering over the gap in the raw note.

## PapersWithCode paper-page routes

A bare PapersWithCode paper URL is an ingest request. Start with route identity, then gather only the resource evidence needed for the selected scope.

Minimal sequence:

1. For `https://paperswithcode.co/paper/<arxiv-id>` and `https://paperswithcode.co/paper/arxiv/<arxiv-id>` shapes, extract `<arxiv-id>` from the URL path with `/(?:paper)/(?:arxiv/)?(\d{4}\.\d{4,5})(?:v\d+)?(?:[/?#]|$)/` as the first route step.
2. Treat the extracted id as sufficient route identity evidence, then immediately resolve canonical paper identity through arXiv metadata; construct `https://arxiv.org/abs/<id>` / `https://arxiv.org/pdf/<id>` for source routing, duplicate search, title/filename selection, and raw-note `source` while later PWC work runs only as resource-route enrichment.
3. Start PWC resource enrichment with a browser-style request wrapper for `https://paperswithcode.co/api/v1/papers/arxiv/<id>?include_resources=true`: set `User-Agent`, `Accept`, and the supplied PWC page as `Referer` on the first API attempt; save compact JSON/HTTP evidence under the task temp root or durable raw-fast reports when the scratch root has already been cleaned.
4. Build one-off PWC repair probes as plain Python functions or JSON/argv-parameterized snippets so URL/header dictionaries remain literal and syntax-safe; write compact artifacts before editing logs.
5. If PWC supplies `url_abs` / `url_pdf` or title metadata, compare it against arXiv metadata. If PWC version is stale and arXiv API resolves a newer latest version, normalize durable source/evidence/ledgers to the arXiv latest version and record the PWC value externally.
6. Extract resource candidates into parse/resource reports only: `repositories[*].url`, `is_official`, `source`, `project_pages[*].url`; normalize nullable arrays to empty lists.
7. Cross-check HF APIs for arXiv-backed papers: `https://huggingface.co/api/papers/<id>` and `https://huggingface.co/api/arxiv/<id>/repos`. Empty PWC arrays are not final if HF exposes exact linked resources.
8. Run exact-link health only for extracted source/PWC/HF/GitHub/project URLs. Preserve redirects/effective URLs externally.

PWC raw-note boundary: raw-note `source` normally uses the canonical arXiv route. `github_links`, `huggingface_model_links`, and `huggingface_dataset_links` may carry only direct source/PWC/HF-API-exposed paper-owned artifact URLs; Hugging Face paper pages, HF search/index pages, HF collections, Spaces, project pages, HTTP/API/probe details, and broader PWC parse details remain external. External parse/resource reports carry PWC/API/GitHub/HF/project URLs, HTTP results, repair artifacts, and parse details. Duplicate patterns must be literal substrings present in the raw note. Preserve compact `evidence_bundle.json` or durable raw-fast report summaries before temp cleanup.

## Artifact boundary table

| Source shape | Default classification | Default action |
|---|---|---|
| Direct first-party GitHub/HF/package/dataset/project route | Source-exposed route, not substance proof | Exact-link health; route/status external only. |
| Code/package/framework route but no model/checkpoint/dataset route | Code/package route only | Exact-link health; do not infer model/data absence. |
| Code/data/release claim without direct route | Claimed artifact unresolved | No title search; artifact existence not inferred. |
| Project page without direct checked GitHub/HF route | Provenance/project route | Optional exact reachability; no crawling or hidden-resource inference. |
| Code Soon/placeholder/disabled/self-linked/roadmap/stale UI | No current direct artifact under default scope | Record unresolved externally; audit only on request. |
| 404/401/403/private/gated/ambiguous exact route | Route unresolved / probe_failed | No fallback search by default. |
| TeX-commented or source-only release URL | Visibility unclear | Exact-link check if plausibly paper-owned; do not promote to confirmed release. |
| Bibliography/dependency/base-model/baseline/cited repo | Not this paper's artifact | Do not health-check as release route unless local context marks ownership. |
| User supplies paper plus explicit artifact URL | User-supplied candidate | Exact-link health only; no broad search or substance audit. |

## First-party, source-bundled, and unresolved substance branches

Use first-party code/package labels only when the checked source or user supplies the route. Outside audit mode, `official_code`, `official_package`, `kernel_or_systems_release`, `config_or_helper_release`, and `code_only_no_model` mean exact route reachability, not license or reproduction completeness. Audit mode may inspect package metadata, source files, launch scripts, configs, hardware/runtime constraints, benchmark harnesses, and whether public artifacts cover training, inference, or microbenchmarks.

Use source-bundled/unresolved branches when e-print packages or official repos expose implementation surfaces but release completeness is unclear:

- E-print/source bundles with `CODE/`, data manifests, CSV/JSON, demos, fixtures, figures, tables, or snippets are source-bundle substance, not necessarily an official GitHub release.
- Direct HF model/card/Space routes get exact-link health by default; HF siblings/checkpoints/configs/card files/collections/Space source require audit scope.
- Public repos with evaluation scripts, prompts, graph/retrieval helpers, benchmark commands, package glue, or framework code are exact-link reachable by default; README commands, license, releases, commits, dependencies, and runtime requirements are audit findings.
- Public visibility does not imply open-source license, full reproduction, available weights, dataset release, or runnable completeness.
- Source bundle / audit workflows need strict provider-token scanning over current source files, note draft, direct-link summaries, and sidecars; ignore generic “token” words without provider-token patterns.
- Figure/table/source-code evidence may support scientific prose, but extracted renders, inventories, and audit details stay in temp/state reports.

## Project-page and no-release branches

Project pages, demos, alpha pages, PWC/CatalyzeX-like pages, mirrors, and family-roadmap repos are not default permission to discover hidden resources. Exact reachability of a paper-owned project page may be recorded as provenance health, but no script inspection, crawling, search, or artifact absence/presence inference follows. In audit mode, parse anchors and classify whether linked routes are current-paper artifacts, prior baselines, dependencies, website source, or unrelated.

## Source-exposed link classification

Before health-checking any URL, read local context. For GitHub/Hugging Face links extracted from paper/PDF text, the default metadata boundary is the paper abstract: abstract links can be direct paper-owned resources; body, reference, appendix, auxiliary TeX/tooling, baseline, and cited-resource links stay in evidence/ignored records unless an explicit platform/API enrichment path proves ownership. Paper-owned examples outside raw text include front-page `Website`, `Code`, `Data`, API/comment code availability, and user-supplied artifact URLs. Non-paper-owned examples: references, baselines, cited datasets, platform chrome, docs/social links, package examples, malformed extraction artifacts. For TeX/e-print, do not trust flat inventories; read surrounding `.tex`, `.bib`, HTML, and PDF text lines. If source contains only TeX/figures/bibliography/style assets, treat it as paper source, not a code release.

## Exact-link evidence storage

`ops.raw_fast_ingest_prepare` writes exact-link evidence storage artifacts for `evidence_report.json/md` and durable closeout reports; `agent_handoff.md` exposes only the body-writing handoff, generated source guidance/fallback locators, sanitized scientific digest, manual-required flag, and script-owned TeX filtered-source refs when the source route produced them. Keep parse reports, health JSON, redirects/effective URLs, API responses, and audit details under `${LLM_WIKI_RAW_FAST_TMP_ROOT}/<slug>/...` during the task. Copy only compact durable summaries to `${LLM_WIKI_STATE_DIR}/raw_fast_reports/*` when needed before cleanup. Never put these artifacts in the human wiki root or raw note. Read manual helper docs only from script-returned `manual_reference_paths` or explicit audit scope.

## Audit-mode expansion

If the user explicitly asks for a resource/reproducibility audit, state scope before probing. Audit mode may inspect repo trees, READMEs, license files, package metadata, HF siblings/cards/checkpoints/datasets/Spaces, project-page anchors, exact-title searches, configs, scripts, launch commands, dependencies, and reproduction boundaries. Still distinguish paper-owned artifacts from dependencies, mirrors, third-party reproductions, umbrella family repos, base models, and prior-work baselines. Do not download large checkpoints/datasets by default.

## Closeout wording patterns

Use automated handoff wording and specialize only with compact paper-route facts: PWC API resolved canonical route; HF APIs cross-checked; source-bundled code/data exists; exact official route reachable/unresolved; project page noted but not crawled; audit-expanded repo inspected. If a manual fallback is required, read only the `manual_reference_paths` returned by the script. Keep all route/resource process wording outside the raw note.
