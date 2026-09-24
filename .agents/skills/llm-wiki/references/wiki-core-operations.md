# LLM Wiki core operations

Load this for ordinary orientation, initialization, ingest, query, lint, archiving, Obsidian integration, risky `_meta`/`log.md` edits, source-count checks, interruption recovery, and workflows that do not need a narrower branch reference.

## Architecture

Default shape:

```text
wiki/
├── SCHEMA.md
├── index.md
├── log.md
├── raw/
├── entities/
├── concepts/
├── comparisons/
└── queries/
```

Layer 1 is raw source material. Layer 2 is compiled wiki synthesis. Layer 3 is `SCHEMA.md`. Production root is `${LLM_WIKI_ROOT}`. Native/generated state stays outside wiki roots under `${LLM_WIKI_STATE_DIR}`.

## Orientation protocol

Before editing or answering from an existing wiki:

1. Read `SCHEMA.md` for domain rules and validation requirements.
2. Read `index.md` for compiled pages and current counts.
3. Read recent `log.md` for recent actions and partial work.
4. Inspect `_meta/topic-map.md` and `_meta/raw-clip-map.md` before creating pages, changing maps, or declaring gaps.
5. Search raw and compiled Markdown for exact source IDs, titles, acronyms, aliases, and canonical slugs.

Always re-read touched files after interruption or preserved-context resume. If the latest visible user message differs from preserved context, reconcile before writing routes, maps, or logs.

## New wiki initialization

Create `SCHEMA.md`, `index.md`, `log.md`, `raw/`, `entities/`, `concepts/`, `comparisons/`, and `queries/`. Write a domain-specific schema, an index with typed sections and one-line summaries, and an initial log entry. Existing clipping roots do not need forced migration; preserve raw source material and keep generated maps in `_meta` or compiled layers, not inside raw source directories.

## Minimal templates

Compiled-page frontmatter:

```yaml
---
title: Page Title
created: YYYY-MM-DD
updated: YYYY-MM-DD HH:MM
type: entity | concept | comparison | query | summary
tags: [schema-tags]
sources: [raw/clip/YYMM/source.md]
---
```

`index.md` should list every compiled page by type and keep a fresh total. Split large sections or rely on `_meta/topic-map.md` for thematic navigation. `log.md` is append-only unless correcting the same action block; rotate only when the schema says the log is too large.

## Ingest workflow

1. Capture raw source first. Use `clip` for ordinary URL/PDF capture and `references/structured-paper-ingest-router.md` for structured papers.
2. Prefer structured Markdown raw notes for paper PDFs. Preserve paper page/PDF routes only where the selected raw-note contract allows them.
3. Search existing raw/compiled pages before writing.
4. Reopen the exact source or canonical raw note immediately before final synthesis.
5. Create/update compiled pages only when they meet `SCHEMA.md` thresholds.
6. Update `index.md`, `_meta` maps, and `log.md` only when the selected route requires it; raw-fast captures may defer this through the pending ledger.
7. Verify before final response and list changed files, checks, and freshness boundaries.

For bulk ingest, read all sources, identify shared entities/concepts/resources, search existing pages once, update pages/navigation once, write one log block, and run the scoped verifier or structural audit appropriate to the batch.

## Query workflow

For ordinary questions, read `index.md`, search raw/compiled Markdown, open relevant pages with `read_file`, and cite the wiki pages used. For comprehensive research questions, also load `references/wikigraph-native-operations-router.md` and fuse raw keyword recall, compiled pages, and native `/query/data` retrieval when graph freshness matters. Write back only for durable comparisons, deep dives, reading paths, diagnostic queries, or coverage audits that would be painful to re-derive.

## Lint, counts, and source refs

Check broken wikilinks, orphan pages, index completeness, frontmatter validity, tag taxonomy, stale pages, contradictions, oversized pages, source refs, and log rotation. Compute headline health over compiled/meta surfaces, not every raw note.

Resolve source refs as external URLs, page-relative `../`/`./`, absolute `/`, or wiki-root-relative `raw/`, `_meta/`, `entities/`, `concepts/`, `comparisons/`, and `queries/`. Ignore documented glob patterns and fenced examples when scanning refs. Do not trust `_meta/raw-clip-map.md` counts blindly during active ingest; verify real filesystem counts and affected chronological batches when counts matter. Use paginated reads for long maps/logs because partial reads can miss entries.

## `_meta`, index, and log edits

Patch long maps using unique filenames, headings, local anchors, or table rows, not generic prose. Re-read the edited region immediately. Confirm both frontmatter `sources:` and intended body/table rows; global filename hits can be false positives. Check table pipe counts, especially when inline math contains `|`; prefer `P(y∣x)` or escaped pipes. `_meta/raw-clip-map.md` and `_meta/topic-map.md` may use different cluster numbering and local anchors.

Re-read `log.md` tail before appending; normalize trailing newlines; verify bullet separation after patching. If only the audit trail is missing after interruption, enrich the existing log block instead of creating a duplicate ingest entry. Strict duplicate checks before logging should resolve only to canonical raw notes; after logging, exact IDs/titles/URLs may also hit `log.md`, so treat those as audit references.

## Sensitive strings, images, and programmatic edits

Scan specifically for Markdown image targets beginning with `data:`; do not flag every prose `data:`. Use strict provider-key boundaries; loose token regexes can match filenames. Never persist credential-looking values from public repos; record only redacted aggregate facts. In Python regex replacements, use `\g<1>` or a function when the replacement begins with digits. After scripted frontmatter edits, re-read the first lines and closing `---` of every touched file.

## Organization, archiving, and Obsidian

For organization/deepening, work by coherent theme cluster: orient, inspect maps, choose a cluster, promote high-value raw sources, expand nearby bridge pages, refresh navigation, validate, and log. Stop creating new concept pages when the cluster is already covered; switch to query-driven synthesis or reading guides.

Archive superseded pages under `_archive/`, remove index entries, replace inbound wikilinks with archived notes, and log the action. Do not silently delete durable knowledge. The wiki can be opened as an Obsidian vault; keep attachments in `raw/assets/` or the configured raw image subtree, and use the Obsidian skill only for Obsidian-specific operations.

## Paper-resource conflicts

Use primary scholarly sources for title/authors/date/abstract/headline results. If README/project/model-card claims conflict with the paper, keep paper-grounded technical prose and record the resource mismatch only as a resource caveat in allowed external surfaces.

## Final response expectations

For ingests/updates, report changed files, verifier results, cleanup, freshness/ledger status, and remaining gaps. For queries, report pages/raw notes actually read. For maintenance, report before/after counts, changed navigation surfaces, and validation output.