# Structured paper ingest router

Use this reference for paper-like sources: arXiv, Hugging Face paper pages, PapersWithCode paper pages, DAIR academy paper pages, OpenReview, DOI/OJS, direct PDFs, GitHub-hosted report PDFs, vendor/lab/project reports, and refreshes of existing structured raw notes. This file is a thin branch selector plus invariant contract; branch references own detailed mechanics.

## Load order

Default structured-paper raw-fast clipping is script-first:

1. Run `python3 -m ops.raw_fast_ingest_prepare --url <source-url>` from `${WIKI_GRAPH_REPO}`.
2. Read the generated `agent_handoff.md` and follow `automation_next_action` / source guidance before opening any advisory sidecars. For TeX handoffs, read the generated filtered TeX source first; `source_read_plan`/`source_tex` are fallback locators only.
3. Stay on the script happy path by default. Load returned `manual_reference_paths` when prepare/handoff reports `manual_required=true`, `resource_review_required=true`, or a concrete `manual_reason`.
4. Use this router as a manual fallback for route identity, existing-canonical refresh, or non-scriptable repair after a generated handoff says it is needed.
5. Use `references/raw-fast-closeout-ledger-contract.md` only when generated closeout args/preview are missing, invalid, or a closeout failure needs repair; otherwise prefer the generated `closeout_args.json` and `closeout_command.preview.sh`.

## Automation/manual boundary

`references/raw-fast-automation-quickstart.md` intentionally stays compact. Generated prepare JSON and `agent_handoff.md` must return `writing_contract_refs` for the note-quality contract before source reads, and return `diagnostic_hint`, `manual_reason`, and exact `manual_reference_paths` when the happy path stops. Use this router and its branch references only after those fields request manual handling; keep route/API/probe/repair details here or in branch references rather than copying them into the quickstart.

## Universal contract

- Identify the paper by canonical title, source id, version, and route set before writing.
- Search existing raw and compiled wiki material by exact title, source id, route URL, DOI/article id, official repo/project URL, acronym, and distinctive method terms before writing.
- Create one canonical raw note first unless the user explicitly requests immediate compiled integration or a query requires it. On the raw-fast happy path, write `raw_body_draft.md`; `assemble_command.preview.sh` writes `raw/clip/<YYMM>/YYMMDDNN_...md` from `candidate_frontmatter.json` plus the body draft. Use clipping/raw-note creation time.
- Keep raw-note frontmatter compact: `title`, `source`, `created`, `updated`, `type`, `domain`, `tags`, `topic_hints`, `github_links`, `huggingface_model_links`, `huggingface_dataset_links`, `capture_route`, and `captured` by default. The link fields carry direct source-exposed paper-owned GitHub/HF model/dataset artifact URLs. `topic_hints` are semantic phrases.
- Raw-note body is scientific reading prose. Route, probe, API, link-health, project/resource audit, and resource-boundary details live in evidence, closeout, status, or log surfaces.
- Formulas/objectives belong inside `## Methodology`; figure/table evidence becomes prose conclusions beside the supported claim.
- Default clipping prioritizes source-grounded reading notes. Direct source-exposed GitHub/HF/package/project artifact routes get exact-link triage; broader resource/reproducibility audits are explicit user-request branches.
- Keep paper-owned artifacts separate from dependencies, mirrors, base models, third-party reproductions, prior-work baselines, platform chrome, and umbrella family repos.

## Branch families

| Paper/resource shape | Load branch reference |
|---|---|
| Plain arXiv/OpenReview paper with no special route or artifact branch | This router + note contract + raw-fast batch + closeout contract. Add `references/arxiv-api-fallback-and-closeout-notes.md` when arXiv API/HTML/source probing stalls or demo/status patches are needed. |
| Hugging Face paper page, PapersWithCode paper page, first-party GitHub/HF/package/project route, unreachable/placeholder route, project-page-only claim, or source-exposed resource ambiguity | `references/structured-paper-source-route-cases.md`; default exact-link triage is handled by `ops.raw_fast_ingest_prepare`. Load manual resource refs only when the user requests an audit or the script returns `manual_reference_paths`. |
| Direct PDF, PDF viewer, DOI/OJS/publisher article, GitHub-hosted report PDF, vendor/lab/project technical report, or report with supplements | `references/structured-paper-non-arxiv-routes.md`; add context-safe long-source when dense. |
| Agentic systems, AI-scientist/self-improvement, benchmark/dataset/evaluation harness, latent looped reasoning, mechanistic theory, proof/theory-heavy cases, or video/3D world-model domain evidence | `references/structured-arxiv-domain-case-bank.md`; also add the source-route branch when source-exposed artifacts are present. |
| Raw note already exists for the same paper/version/route | Use the existing-canonical refresh section below, then close out through `references/raw-fast-closeout-ledger-contract.md`. |
| Multiple supplied routes may identify different papers | Resolve route identity below before selecting a branch; separate notes when identity differs. |

## Route identity and duplicates

Treat every supplied route as provisional until identity is proven. This applies to arXiv IDs, conference pages, OpenReview, DOI pages, project pages, GitHub/HF routes, PWC/HF paper pages, direct PDFs, and local PDFs.

For mismatched multi-route inputs:

1. Probe each route enough to establish title/authors/abstract/method identity: arXiv API/abs/PDF/HTML/e-print, DOI/publisher metadata, conference/OpenAccess metadata, OpenReview metadata/PDF, PDF text, local PDF metadata, project page, or source-exposed GitHub/HF metadata.
2. Search the wiki for every candidate identifier and title before writing.
3. If routes are distinct papers, write separate canonical raw notes and close each out separately. Keep route-mismatch evidence in temp evidence/closeout/status surfaces, not raw-note frontmatter/body.
4. Keep resources scoped to the matched paper; never reuse a repo/HF/project route across notes because it appeared in the same browsing session.

For existing canonical raw notes:

1. Prove same-paper identity by exact title/source id and by reading the candidate raw note and recent log entries. Roundup/newsletter/bibliography hits are not canonical duplicates.
2. Re-check current source surfaces as available. Patch in place only when durable facts changed, the note predates the current raw-note contract, or the user asked for refresh; preserve filename and original `created`/`captured` history.
3. Remove legacy `resource_status`, route fields, authors/categories, version fields, checksums, probe facts, route/probe prose, top-of-note metadata blocks, or old reproduction caveats from the raw note and keep those facts externally.
4. If the existing note is already current and source-clean, report no-write canonical confirmation. If durable text changed, use the existing-refresh closeout exception in `references/raw-fast-closeout-ledger-contract.md` and do not create a duplicate ingest.

## Route-specific reminders

- OpenReview forum/PDF/attachment links are handled by `ops.raw_fast_ingest_prepare`: it normalizes `forum?id=...` and `attachment?id=...&name=pdf` to the canonical `https://openreview.net/pdf?id=<id>` source, reads the credential-file location from `OPENREVIEW_ENV_PATH` in the repository's ignored `.env` (credentials stay in that separate file and are read inside Python), and uses `openreview.api.OpenReviewClient(baseurl='https://api2.openreview.net')` to fetch metadata/PDF into temp evidence. Keep OpenReview API/auth/probe details in temp evidence/closeout surfaces, do not print/source credentials in shell output, and only switch to manual OpenReview handling if the prepare JSON or generated handoff returns a manual stop.
- Hugging Face paper pages are input/discovery routes, not proof that HF-owned artifacts exist. Resolve canonical arXiv/source identity, inspect paper-owned links in arXiv/HTML/e-print/source/API comments, and run exact-link health only for source-exposed or user-supplied routes.
- PapersWithCode paper pages are discovery routes. Use `references/structured-paper-source-route-cases.md` to resolve PWC API paper identity and resource arrays, cross-check HF APIs when arXiv-backed, and keep PWC/GitHub/HF parse details outside the raw note.
- DAIR academy paper pages (`academy.dair.ai/papers/<slug>-<arxiv-id>`) are discovery routes: `ops.raw_fast_ingest_prepare` resolves the trailing arXiv id from the slug and reads canonical arXiv surfaces, keeping the DAIR URL only as external supplied-route provenance. Non-paper DAIR pages (index/collections/week/hero) carry no trailing id and are not routed.
- If a user supplies a DOI/venue alongside an arXiv paper, keep arXiv as canonical raw-note `source` when it is the evidence route; cheap exact DOI/venue health belongs only in external evidence/closeout/report surfaces.
- Duplicate strict patterns should be literal substrings that occur in the raw note: canonical source id, exact title, and a paper-owned term/URL only when it deliberately appears in the note. Do not use umbrella framework repo URLs as the only strict duplicate pattern.

## Final handoff

After note synthesis, run the generated `closeout_command.preview.sh` / `closeout_args.json` when available; open `references/raw-fast-closeout-ledger-contract.md` only if those artifacts are missing, invalid, or the closeout fails. Report only verified status: raw saved or refreshed, verifier result, wiki queue/action, temp cleanup, graph blocker or graph queue/action, validation/report paths, and unresolved resource caveats. Never imply compiled/wiki or native graph freshness when raw-fast wiki integration or native refresh remains pending.
