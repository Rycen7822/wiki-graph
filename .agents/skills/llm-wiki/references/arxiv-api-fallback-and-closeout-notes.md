# arXiv API fallback and raw-fast closeout notes

Use this as a compact playbook when clipping bare arXiv paper links into llm-wiki and the usual metadata path is slow or ambiguous.

## Metadata resolution

- Prefer the arXiv API for canonical metadata when you need a stable title/version/summary quickly:

```bash
curl -L --connect-timeout 20 --max-time 60 \
  'https://export.arxiv.org/api/query?id_list=<arxiv_id>'
```

- Use the API entry to recover the authoritative title, authors, versioned abs/pdf links, category, published time, and comments/project-page hints.
- If a Python `urllib` probe stalls, switch to `curl` rather than retrying the same path with longer waits.
- When a script will parse the arXiv API response as XML, call curl in silent-error mode, e.g. `curl -sS -L --connect-timeout 20 --max-time 60 ...`. Some command wrappers can capture a non-silent curl progress meter before the XML and produce `ParseError: line 1, column 0`; if that happens, inspect the first bytes and refetch with `-sS` before treating the API as unavailable.

## Raw-fast clipping notes

- For default clipping, a bare arXiv paper link should be treated as structured paper ingest.
- When the arXiv API is rate-limited but the abs page is healthy, recover the version from abs metadata (`og:url`, versioned HTML link, submission-history version row) and use that versioned abs route for raw-note `source`, closeout `source_id`, and duplicate patterns. Do not wait on repeated API retries if PDF/HTML/e-print/abs evidence is already available.
- Raw-fast automation owns source-exposed resource links: it scans abs metadata comments and TeX title/author blocks for `Code:` / `Data:` routes, verifies exact-link health, and adds reachable GitHub/Hugging Face model/dataset routes to script-owned metadata.
- If the source does not expose direct GitHub/Hugging Face artifact links, leave resource metadata empty and continue the raw-note flow.
- Use the versioned arXiv id and exact API/abs title as the strict duplicate patterns when closing out the raw note.

## HTML route and unresolved demo/status patches

Use this subsection when a structured arXiv ingest needs explicit API/abs/PDF/HTML/source coverage, or when paper/source mentions a website, GUI, demo HTML, supplementary interface, or project page but no public artifact route is obvious.

- Probe arXiv HTML separately from API/abs/PDF/e-print/source; an evidence bundle that confirms API/abs/PDF/source does not prove `https://arxiv.org/html/<id>` availability. Check latest and versioned routes when useful, such as `/html/<id>` and `/html/<id>vN`, and record `200` or unavailable status externally.
- Do not turn source-only demo mentions into verified releases. If source/supplement mentions files such as `index_static.html`, `qualitative.html`, “website we provided”, GUI, demo, or project page, verify that the files are actually in the e-print archive or a reachable source-exposed URL exists. If neither exists, write `source-mentioned supplementary GUI/website unresolved`, not `public demo verified`.
- Keep dependency/protocol links separate from paper-owned artifacts: ORCID, arXiv links, Prolific/media-source links, cited datasets, and platform/chrome links are not code/data/model releases unless the paper/project explicitly owns and publishes artifacts there.
- For no-release cases, inspect direct source links only. If there is no direct GitHub/HF URL, record `not_checked` by default; if there is a direct route, run only exact-link health and treat endpoint/network failures as `probe_failed`, not verified absence.
- If HTML/resource status is corrected after closeout, update the external evidence/closeout report and compact `log.md` resource line, then rerun `${WIKI_GRAPH_REPO}/.agents/skills/llm-wiki/scripts/raw_fast_note_verify.py` using a wiki-root-relative `--raw-file`. Do not add `resource_status` frontmatter or a standalone resource-status/raw-note process section.

Example verifier form after a late status patch:

```bash
python3 ${WIKI_GRAPH_REPO}/.agents/skills/llm-wiki/scripts/raw_fast_note_verify.py \\
  --wiki ${LLM_WIKI_ROOT} \
  --raw-file raw/clip/<YYMM>/<note>.md \
  --structured-paper \
  --patterns '<arxiv-id>' '<method-name>' '<exact-title>'
```

## Closeout hygiene

- After rewriting a raw note, re-open the written file and scan for stray ASCII control characters before closeout.
- If a patch or copy step introduces an accidental control byte, repair the surrounding prose and verify the repaired line again before running closeout.
- Keep temporary evidence under `${LLM_WIKI_RAW_FAST_TMP_ROOT}/<slug>/`; do not leave arXiv PDFs, source bundles, or extraction artifacts elsewhere.
