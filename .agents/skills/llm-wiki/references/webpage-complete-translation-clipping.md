# Webpage complete clipping + Chinese translation

Use this reference when the user asks to fully clip a non-paper webpage/blog/static article and include a complete Chinese translation. Use `references/webpage-obclip-clipping.md` for browser/obclip mechanics when the capture route needs it.

## Trigger

- User asks for 完整剪藏 / full clip / full original + translation / 翻译成中文.
- Source is an ordinary technical blog, Substack-style article, company/static article, Hugging Face Space static blog, or similar webpage.
- The task is not a structured paper raw-fast route. If the page is paper-like or exposes a direct paper/PDF route, route through the paper/raw-fast contract first unless the user explicitly asks for ordinary webpage clipping.

## Route

1. Deduplicate by exact title, slug, canonical source URL, author/domain, and stable source-page URL across `raw/clip`, `_meta`, and `log.md`.
2. Capture the article body with the simplest successful route: obclip, browser/static extraction, or rendered DOM fallback. Inspect the saved body before accepting the capture.
3. Cross-check metadata from the live page or static HTML: title, description, author, publish date, canonical URL, og URL, and any advertised PDF or resource links.
4. Write or update the canonical raw note under `raw/clip/<YYMM>/<YYMMDDNN>_<readable-title>.md`.
5. Preserve the full source body first, then append a complete Chinese translation under a separate H2.
6. Localize retained images and verify no remote/data image links remain.
7. Mark pending wiki integration unless immediate integration is explicitly requested or the threshold/pre-query flow requires it.

## Raw note shape

Recommended H2 structure:

```markdown
## 英文原文（完整剪藏）
...

## 中文翻译（完整）
...
```

Frontmatter should carry the durable source metadata allowed by the current `llm-wiki` contract: title, source/source_page when needed, created, updated with `HH:MM`, type, domain, tags, topic_hints, capture_route, captured, translation status, and image-localization status. Keep route probes, checksums, browser errors, link-health, API details, and resource audit facts out of the raw body unless the current task explicitly asks for an audit surface.

Body constraints still apply:

- Do not paste bare URLs, markdown links, process evidence headings, link-health details, API/probe facts, or resource-boundary narration into the raw-note body.
- If the original article contains many links, preserve the visible anchor text where semantically useful and keep the actual URLs in metadata/evidence/log only when needed.
- Preserve headings, paragraphs, lists, code blocks, tables, formulas, captions, and figure placement as content.
- Remove page chrome, legal boilerplate, subscription prompts, promotional/course ads, nav/footer blocks, and unrelated site links.

## Translation workflow

- Split long articles by stable headings or line ranges from the canonical localized source body.
- Translate explanations, not identifiers that are clearer in English.
- Preserve Markdown structure, heading hierarchy, code fences, tables, formulas, figure positions, and captions.
- For technical terms, prefer Chinese plus English on first mention when helpful, such as `KV cache（KV 缓存）` or `cross-layer attention（跨层注意力）`.
- Recombine under `## 中文翻译（完整）` and remove placeholders.
- Verify key headings from the source are mirrored or intentionally translated in the Chinese section.
- Do not replace the source body with the translation; complete clipping + translation keeps both.

## Images

- Extract remote markdown images from the accepted source body.
- Download each retained image to `raw/images/<doc-folder>/`.
- Rewrite image references in the English original to relative local paths such as `../images/<doc-folder>/imgNN.<ext>`.
- Let the Chinese translation reuse the same localized image paths when preserving figure positions.
- If inline SVGs are small explanatory diagrams rather than external figures, convert them into concise textual equivalents that preserve semantic labels/tokens instead of keeping a huge raw SVG blob.
- Final image check: remote markdown images = 0, `data:image` = 0, missing local image files = 0.

## Hugging Face Space / static blog nuances

- Preserve both the requested anchored URL and the de-anchored source page in allowed metadata when useful.
- Some static pages expose a clean PDF route; fetch/check it as a cross-check when advertised, but do not treat it as the only source if HTML extraction succeeds.
- Some pages advertise localhost canonical/og URLs; do not use those as canonical when the live user URL is clearly the source.
- Preserve code blocks, tables, model names, formulas, anchor headings, and technical identifiers such as method names or loss-mask terms.

## Substack / ordinary blog nuances

- Try obclip first for ordinary pages, usually with `--settle-ms 5000`, then inspect the output.
- If an over-strict selector times out, retry once without it before switching to manual reconstruction.
- Cross-check HTML metadata with a browser-like fetch or browser snapshot when the route needs provenance.
- For source pages with book/promo notes that are part of the article body, keep substantive author notes but remove generic subscription/paywall/footer chrome.

## Integration policy

- Default closeout for ordinary webpage clips is raw saved + pending wiki integration.
- Update `_meta`, compiled pages, `log.md`, or native/zvec state only when immediate integration is explicitly requested or the threshold/pre-query flow runs.
- Never claim native graph freshness while wiki integration is pending.

## Verification checklist

- Canonical raw note exists at the expected path and duplicate search resolves only to it.
- English original and complete Chinese translation both exist.
- Raw body contains no bare URLs, markdown links, process evidence sections, remote markdown images, or `data:image` payloads.
- Local image references resolve and reused translated-section image paths point to the same localized files.
- Exact source/title/key phrase checks pass against the canonical note.
- Changed wiki/meta/log pages, if any, have resolving wikilinks and relative source refs.
- Temp capture, translation chunks, scripts, and downloaded scratch files are removed before final reporting.
