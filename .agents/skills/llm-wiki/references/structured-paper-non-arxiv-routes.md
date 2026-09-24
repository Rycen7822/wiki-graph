# Structured paper non-arXiv routes

Use this aggregate reference for structured raw-fast paper clipping when the evidence route is a DOI/OJS/publisher article, direct PDF, PDF viewer, vendor/lab/project technical report, GitHub-hosted report PDF, local PDF, or long publisher report with supplements rather than a normal arXiv/OpenReview route. Pair with `references/structured-paper-note-contract.md`; use `references/context-safe-long-source-clipping.md` for dense/long sources; let `ops.raw_fast_ingest_prepare` handle resource URLs by default and read manual resource docs only from script-returned `manual_reference_paths` or explicit audit scope; close out through `references/raw-fast-closeout-ledger-contract.md`.

## Trigger and route choice

Use this file when the user supplies or the source resolves to:

- Direct PDF URL, PDF.js/viewer URL, GitHub blob/raw PDF, vendor/lab/project report PDF, project technical report, or local PDF without stable arXiv/OpenReview/DOI landing route.
- DOI, OJS/proceedings article page, publisher landing page, publisher PDF/view/download route, or Nature/Science-style article with main PDF plus supplements.
- Short position/agenda/workshop paper where conceptual framing, checked absence of formulas/figures/tables/artifacts, or testable research-program claims matter.
- Long technical report where engineering detail, supplements, prompts/pseudocode, reporting summaries, access/safety limitations, or route mismatch must be preserved.
- Multiple supplied routes where arXiv/local PDF/project/resource URL may identify a different paper.

Choose one canonical raw-note `source`: normally the user-supplied stable DOI/article/landing/PDF route. Auxiliary PDF/raw routes, DOI/Crossref/publisher metadata, supplementary URLs, local attachment paths/checksums, hashes, page counts, and mismatch evidence stay in temp evidence, reports, closeout, ledgers, or compact `log.md`, not raw-note frontmatter/body.

## Identity and route-resolution workflow

1. Resolve identity independently across the user route, landing/article page, PDF route, DOI/Crossref/publisher metadata, local attachment, and any supplied arXiv/project/resource route.
2. Record title, source id/DOI/article id, authors, venue/year, canonical route, supplied route, PDF route, and mismatch decisions externally. Raw frontmatter stays minimal; do not add `source_pdf`, `doi`, `pdf_url`, `supplement_url`, checksums, page counts, authors, venue, or resource-status fields.
3. If a supplied arXiv/local PDF route has different title/authors/abstract/DOI, preserve `route_mismatch` externally and exclude it from synthesis. Do not merge unrelated arXiv metadata into a DOI/direct-PDF note.
4. For PDF.js/viewer/project landing routes, inspect HTML title, meta tags, citation PDF/fulltext tags, OpenGraph data, scripts, robots/meta description, and relative PDF paths enough to find the real PDF while keeping the stable supplied/article route as canonical source when appropriate.
5. For GitHub-hosted PDFs, compare blob/raw routes. Use raw URLs for extraction/evidence when helpful, but keep the user-supplied blob/source route as canonical `source` when that was the input route. Do not promote the hosting repo to an implementation artifact unless the report explicitly labels it as Code/Data/Model/etc. for this paper.
6. If Docling or another extractor guesses a nonsense title, recover title from first real Markdown heading, first-page text, PDF metadata, DOI/publisher metadata, or GitHub path.

## Evidence extraction

- Download PDFs with headers when needed and keep PDF, text, metadata, hash, bytes, page count, outline/page map, extraction status, and link inventories under `${LLM_WIKI_RAW_FAST_TMP_ROOT}/<slug>/...` only. Large image-heavy PDFs can be legitimate; if the script returns a `DownloadTooLarge` / `FileTooLarge` stop, treat the returned `diagnostic_hint` as the owner of retry flags such as `--max-download-bytes 4GiB` or `--max-download-bytes none` for trusted sources, and keep size/probe facts outside the raw note.
- For long GitHub-hosted surveys/reports, PyMuPDF page-sliced extraction is acceptable as a temporary fallback for page text, metadata, outline, link annotations, and caption-like figure/table candidates; do not add persistent dependencies just for this.
- For long reports, dense appendices, supplements, or explicit “don’t lose details” requests, follow `references/context-safe-long-source-clipping.md`: inventory scope/size, split source slices by density, write scratch evidence after each batch, draft incrementally, reopen source for exact claims, reread full draft, record confidence review, then remove temp root.
- Supplementary files can contain prompts, pseudocode, methods, benchmark rubrics, reporting summaries, safety/access policies, implementation constraints, peer-review material, or unrelated platform files. Classify each by link text, filename, and document title before using it.
- For position/conceptual papers, check formulas/figures/tables through extracted text cues and, when useful, PDF object/image probes. If absent, treat checked absence as evidence about claim strength rather than a missing-work gap.

## Resource and artifact boundary

Default non-arXiv clipping is still not a resource/reproducibility audit. Extract annotations/text links for arXiv IDs, DOI strings, GitHub/HF routes, project pages, dataset pages, licenses, code/data/model mentions, and artifact buttons, then classify local context before probing.

Paper-owned cues include title-page/front-page Website/Code/Data/Model/Benchmark/Release links, explicit code-availability text, source-exposed project-page buttons for the same paper, and user-supplied resource URLs during the same clipping. Non-artifact cues include bibliography/reference links, cited dependencies, baseline repos, unrelated host repository files, website source assets, model/base dependencies, project-page chrome, and malformed extraction artifacts. By default, health-check only exact source-exposed or user-supplied artifact routes; never search GitHub/HF/project pages by title, method, DOI, arXiv id, author, lab, repo owner, or acronym to fill missing resources.

If a direct route fails or disagrees with a source-exposed canonical route, record `probe_failed` / unresolved externally and do not “fix” it through broad search unless the user asks for an audit. Scan strict secrets only over current per-task temp/evidence plus the target raw note, excluding binaries/model weights.

## Note-writing rules

- Use the structured paper sections from `references/structured-paper-note-contract.md`.
- Put formulas/objectives/pseudocode in `## Methodology`; put figure/table/benchmark/wet-lab evidence beside the supported method/result/limitation claim.
- Do not add standalone formula/figure/resource/process headings, embed images, copy charts, reproduce large tables, or write bare URLs/Markdown links in the raw-note body.
- Frontmatter stays minimal. Only the canonical source route goes in `source`; use concise `capture_route` values such as `direct_pdf`, `publisher_doi`, `ojs_article`, `publisher_pdf`, `github_hosted_report`, or `vendor_technical_report`.
- For long reports, preserve reusable engineering detail in prose: training-scale numbers, data-governance/drop-order/filtering logic, architecture modules, pipeline state, prompts/pseudocode, ablations, evaluation protocol, benchmark construction, human/expert review, wet-lab settings, runtime/fault tolerance, safety/access limits, and conceptual-vs-runnable reproducibility boundaries.
- For position/agenda papers, classify contribution honestly as a research agenda, conceptual framework, measurement program, hypothesis paper, or position paper. Do not inflate conceptual framing into empirical results. If formulas/figures/tables/artifacts are absent, state checked absence only where it affects Methodology/Results/Limitations.

## Duplicate checks and closeout

Search existing raw and compiled wiki entries before writing using exact title, DOI/article id, DOI URL, article/view URL, stable PDF URL/path, canonical publisher route, local attachment checksum/path when relevant, matching arXiv id only after identity confirmation, and distinctive paper-specific entities. Do not use unrelated supplied routes or broad framework/repo URLs as strict duplicate patterns.

If an exact canonical raw note exists, refresh it instead of creating a duplicate; otherwise write one canonical raw-fast note under `raw/clip/<YYMM>/YYMMDDNN_<readable-title>.md` using clipping/raw-note creation date semantics.

Use `references/raw-fast-closeout-ledger-contract.md` for verifier, duplicate-count interpretation, pending ledger, temp cleanup, wiki-integration threshold, and native graph gate. If pending count reaches threshold, run the current auto-integration path; otherwise report raw saved plus pending count and do not claim compiled/wiki or native graph freshness. Final response includes only verified status: note path, verifier result, artifact/resource boundary summary, wiki validation or pending count, threshold action, temp cleanup, and native post-status only when graph refresh actually ran.
