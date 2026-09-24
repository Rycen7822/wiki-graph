# Raw-fast resource probe boundaries

Use this reference when a paper, blog, article, project page, or user message surfaces GitHub, Hugging Face, package, project, dataset, demo, or other artifact URLs during llm-wiki clipping. It owns the default direct-link-only policy, raw-note wording prohibition, conservative labels, and exact-link helper command. Structured-paper route classification lives in `references/structured-paper-source-route-cases.md`; note structure lives in `references/structured-paper-note-contract.md`.

## Default rule

Default clipping prioritizes source-grounded reading notes. Reproducibility audits are explicit user-request branches.

- If the checked source does not directly contain a GitHub/HF/package/project artifact URL, do not search by arXiv ID, DOI, title, acronym, method, author, lab, repo-owner guess, dependency name, or project-page scripts. Report `not_checked` externally rather than `verified_absent`.
- If the source directly contains an artifact URL, verify exact-link health only: HTTP status, redirect/effective URL, content type, public/gated/private/unreachable status, and cheap API/page status for that exact route when unambiguous.
- If the user supplies a paper/source plus an explicit GitHub/HF/package URL, treat the URL as a user-supplied candidate route. Run exact-link health only, preserve the exact URL in external evidence/closeout, and phrase scope as user-supplied exact-link reachable/unresolved with no substance audit.
- Project pages, demo pages, paper mirrors, PWC/alphaXiv/CatalyzeX-style pages, and bibliography/dependency links are provenance/discovery surfaces, not default permission to crawl hidden artifacts. A cheap exact reachability check of a paper-owned project page is allowed as provenance health, but no page crawling, hidden-resource discovery, repo/HF search, or artifact absence/presence inference follows by default.
- Ignore platform/chrome URLs introduced by rendering surfaces, such as arXiv HTML feedback GitHub links, LaTeXML issue links, Hugging Face documentation/social links, widgets, recommendation links, and malformed extraction artifacts.
- Classify every extracted URL by local context before probing. Paper-owned cues include front/title-block `Website` / `Code` / `Data` / `Model` links, explicit method-release statements, arXiv API/comment metadata such as “Code is available at ...”, and TeX/PDF abstract/title-block release links. Non-paper-owned cues include reference-list entries, baseline repos, cited datasets, dependency URLs, documentation links, platform chrome, and package examples.
- Flat URL inventories are not authority. PDF extraction, `links.json`, `resource_probe.json`, and `*.urls.txt` can mix first-party artifacts with cited tools and can lose wrapped source URLs. Read surrounding PDF/HTML/TeX/BibTeX lines around each GitHub/HF/package candidate before deciding.
- Line-wrapped and source-commented routes need caveats. A paper-body or source footnote such as `All resources publicly available at \url{...}` is paper-owned even if split across lines. A commented-out or source-only route may be exact-link checked when plausibly paper-owned, but record visibility separately from reachability and do not promote it to a confirmed release.
- Broaden probes only when the user explicitly asks for a resource/reproducibility audit; state the expanded scope before probing and keep large checkpoint/dataset downloads out of default clipping.

## Raw-note wording prohibition

Raw notes are source-clean reading notes. Do not add resource-status frontmatter, route/probe metadata, Markdown resource links, link-health/API facts, project/GitHub/HF/PWC details, repo/license/file-count/runtime audit facts, resource-boundary paragraphs, “link reachable but not audited” caveats, or standalone process/resource headings. The only raw-note resource URLs allowed by default are direct source-exposed paper-owned `github_links`, `huggingface_model_links`, and `huggingface_dataset_links`; keep all other resource facts in task temp evidence, raw-fast reports, closeout reports, pending ledgers, compact `log.md`, or final response caveats.

## Conservative labels

Use these labels only in external evidence, reports, ledgers, closeout, compact `log.md`, or final status:

- `source_link_reachable`: an exact source-exposed or user-supplied route returned successful status or stable redirect.
- `source_link_unreachable`: an exact route returned 404/403/401/network failure/rate-limit/ambiguous redirect; unresolved, not absence elsewhere.
- `probe_failed`: exact route could not be confirmed at capture time.
- `not_checked`: no direct artifact route was present, or audit was not requested.
- `dependency_not_artifact`: cited/base/dependency resource, not this paper/source's release route.
- `audit_requested`: the user explicitly asked for broader reproducibility/resource audit.

A reachable route verifies reachability only. It does not verify license, repo substance, checkpoint/config/weight files, dataset splits, collection members, reproducibility, or runnable completeness.

## Exact-link helper

Use the reusable direct-link helper for exact-link health checks:

```bash
python3 ${WIKI_GRAPH_REPO}/.agents/skills/llm-wiki/scripts/direct_link_health_check.py \\
  'https://github.com/owner/repo' \
  'https://huggingface.co/namespace/resource' \
  > ${LLM_WIKI_RAW_FAST_TMP_ROOT}/<slug>/direct_link_health.json
```

The helper uses short curl timeouts, follows redirects, tries `HEAD` before capped `GET`, runs checks in parallel, and emits JSON with status, effective URL, content type, timing, and bounded errors. URLs are positional arguments; do not invent `--url` flags. If customization is needed, use actual helper flags such as `--connect-timeout`, `--head-max-time`, `--get-max-time`, and `--retry`.

If the helper exits nonzero for an unreachable exact route but writes JSON, read the report and continue clipping unless the user's audit depends on that route. Record the route as unresolved/probe_failed rather than broad-searching replacements.

Allowed by default:

- `HEAD` or capped small `GET` for exact linked URLs when needed.
- GitHub repo API status for the exact `owner/repo` only, if cheaper than page rendering.
- HF model/dataset/Space API status for the exact namespace/type only when the linked URL unambiguously identifies one.
- Exact project-page reachability as provenance health when the page itself is source-exposed.

Not allowed by default:

- GitHub/HF/package/project search APIs using title/arXiv/DOI/method/author/acronym terms.
- Project-page crawling to discover hidden artifact links.
- README/tree/license/commit/release/package inspection.
- HF siblings/card/checkpoint/dataset split/Space source enumeration.
- Classifying “no official code/model/dataset” from skipped searches.

## External closeout wording patterns

- No direct route: `checked source surfaces; no direct paper-owned GitHub/HF/package/project route found; broad resource discovery not run by default; strict_secret_hits=0`.
- Direct route reachable: `source-exposed exact artifact route reachable; no repo/HF/package substance, license, data/model, or reproduction audit run by default; strict_secret_hits=0`.
- Direct route failed: `source-exposed exact route returned <status/error> and is unresolved; no broad fallback search run by default; strict_secret_hits=0`.
- Project page only: `source-exposed project page noted/reachable as provenance; project page not crawled and hidden artifact discovery not run by default; strict_secret_hits=0`.
- User-supplied route: `user-supplied exact route reachable/unresolved; no broad search or substance audit run by default; strict_secret_hits=0`.
