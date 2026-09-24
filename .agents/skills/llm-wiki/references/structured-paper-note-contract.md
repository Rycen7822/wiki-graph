# Structured paper note contract

Use this reference whenever a paper-like source is clipped into llm-wiki as a structured raw note. It owns raw-note shape, semantic quality, integrated formula/figure/table evidence, body/frontmatter boundaries, and closeout-facing constraints. Use generated `agent_handoff.md` for task-specific source paths and `references/raw-fast-closeout-ledger-contract.md` for exact verifier/ledger commands. Read manual route/resource references only when automation returns `manual_reference_paths`.

## Required raw-note shape

Structured paper raw notes use this canonical H2 sequence:

1. `## 一句话总结`
2. `## 论文摘要（中文）`
3. `## Motivation`
4. `## Methodology`
5. `## 关键实验结果 / 作者结论`
6. `## 对未来研究的启发`
7. `## 可能的局限`
8. `## 可继续追问的问题`

Do not add standalone process/resource/evidence buckets such as `## Evidence trail`, `#Evidencetrai1`, `## 资源与复现状态`, `## 关键公式 / 机制推导`, or `## 关键图表 / 读图笔记`. Scratch aliases are fine; canonical raw notes keep the exact headings above.

## Raw-note quality bar before closeout

The verifier is a structural/safety gate, not a quality oracle. Before assembly/closeout, the clipping item owner—a single-paper agent or the responsible batch worker—must reread the body as a human reading note and repair shallow sections instead of adding keywords. In batch mode this semantic reread and repair remain inside the item worker; the parent coordinator does not repeat them. A good structured paper note lets a future reader answer: what problem/tension the source attacks, what mechanism/protocol it introduces, what evidence supports the claim, where the conclusion is bounded, what reusable insight remains, and what follow-up questions matter.

Use the generated handoff as the owner of source paths. Read `paper_digest.md` as an evidence index, not as a replacement for the source. When the handoff exposes TeX source refs, read `tex_agent_ir_audit.md` and `paper_source.agent.tex` as the filtered original-TeX source surface. `source_read_plan.first_reads` and `source_tex` are fallback locators only after the filtered TeX leaves a named exact-span gap: formula details, table/figure conclusions, limitations, or uncertain claims. For non-TeX/PDF/HTML routes without filtered TeX, use generated source spans wherever claims need more evidence. Clipping quality wins; fallback is gap-driven.

When key figures/tables are local images or figure PDFs and the note relies on their trends, render or visually inspect the actual evidence when possible. If vision is unavailable or unreadable, triangulate from TeX captions, table rows, surrounding prose, HTML/PDF structure, and source text; state the evidence surface honestly.

Section jobs:

- `## 一句话总结`: one dense thesis sentence/short paragraph naming object, mechanism/lens, evidence/outcome, and trade-off; avoid generic “提出了一种方法”.
- `## 论文摘要（中文）`: compress the paper into a small model: problem framing, method, evaluation/theory, headline conclusion, and important caveat; do not translate the official abstract sentence by sentence.
- `## Motivation`: explain source-specific tension/gap, what prior framing misses, and why the question is nontrivial.
- `## Methodology`: mechanism center. Preserve objects, state variables, data/control flow, equations, update rules, protocol, module roles, taxonomy axes, benchmark construction, or system state/artifact flow.
- `## 关键实验结果 / 作者结论`: turn figures/tables/benchmarks/proofs into bounded conclusions with task/metric boundaries, compact numbers, deltas, rankings, ablation directions, qualitative panel takeaways, or failure cases.
- `## 对未来研究的启发`: extract reusable design/research principles: interfaces, ablations, architecture knobs, verifier/provenance requirements, deployment constraints, evaluation dimensions.
- `## 可能的局限`: name concrete limits from scope, data, assumptions, implementation, metric, benchmark, evidence surface, access, or generalization path.
- `## 可继续追问的问题`: ask targeted questions that can drive future wiki search, synthesis, or experiments; prefer 5--10 concrete questions over generic curiosity.

## Minimum completeness floor

This is an anti-shallow closeout floor, not a padding quota. For a normal accessible paper, do not accept a draft where each H2 is one tiny paragraph or generic bullets. If the source is genuinely short, abstract-only, access-limited, or a narrow announcement, state that boundary and still preserve the best mechanism/evidence.

- `## 一句话总结`: problem + mechanism/lens + result/evidence + trade-off.
- `## 论文摘要（中文）`: normally 2--4 substantive paragraphs, or one dense paragraph only for narrow sources; cover problem, method, evidence, conclusion, and visible boundary.
- `## Motivation`: at least two source-specific gap/tension points; generic “LLM 很重要 / benchmark 很重要” is not enough.
- `## Methodology`: normally at least three substantive paragraphs or an equivalent dense walkthrough. Name central objects/state, pipeline/protocol order, equations/schema/taxonomy/stages when available, and why the design addresses the motivation. Benchmark papers need task construction, labels/metrics, negative/adversarial cases, and evaluation protocol; systems papers need modules, state/artifact flow, gates, cost/provenance boundaries.
- `## 关键实验结果 / 作者结论`: normally at least three evidence claims. Each should tie a Table/Figure/benchmark/ablation/proof result to a bounded conclusion with compact values, metric names, task boundaries, or qualitative panel facts when available.
- `## 对未来研究的启发`: normally 4--8 reusable implications or rich paragraphs, each mapped to a design/evaluation/ablation/provenance/deployment decision.
- `## 可能的局限`: normally at least three source-specific limitations; avoid “需要更多实验” unless it names the missing evidence surface.
- `## 可继续追问的问题`: normally 5--10 targeted questions, concrete enough to drive future reading or experiments.

Length is evidence-dependent, not capped. Narrow complete notes may land around 3k--4k body characters; ordinary technical papers often need 6k--10k; dense reports, surveys, theory/mechanistic papers, subtle engineering notes, or source-heavy papers can be longer and may need the long-source scratch workflow. A normal accessible paper below roughly 3k body characters is suspicious unless the source boundary justifies it.

Closeout self-test: before verifier/closeout, the clipping item owner rereads the full body and asks whether a future reader can explain the core problem, mechanism/protocol, key evidence/metrics, caveats, reusable insight, and follow-up questions without reopening the source. If not, that same owner reopens handoff/source spans and expands weak sections. In a batch, this self-test must finish before the worker emits `ready`; the parent coordinator relies on the script-owned result and does not perform a second manual note read.

## Density and style

- Prefer analytical prose. Bullets are fine for implications, limitations, and questions, but each bullet should carry a cause/effect, decision boundary, or concrete research use.
- Keep formula, figure, table, protocol, and limitation evidence beside the claim it supports; do not create separate evidence buckets.
- Verifier failures should trigger semantic repair, not keyword stuffing. If a shallow note passes structurally, repair it anyway.
- Use domain-specific structure: surveys preserve taxonomy axes/trade-offs; theory/mechanistic papers preserve assumptions, variables, proof/geometry skeletons, and caveats; benchmark papers preserve task construction, metrics, negative cases, and failure modes; systems/agent papers preserve pipeline state, artifact/provenance boundaries, costs, and deployment constraints; optimizer/training papers preserve simplified model, update rule, generalization consequence, and scale-transfer caveat.

## Frontmatter and body contract

Default frontmatter keys are `title`, `source`, `created`, `updated`, `type`, `domain`, `tags`, `topic_hints`, `github_links`, `huggingface_model_links`, `huggingface_dataset_links`, `capture_route`, and `captured`. Use `domain` for semantic area (`machine-learning`, `alignment`, `robotics`, `computer-vision`, etc.), not source form; new raw-fast paper notes should not use `domain: paper`. Keep GitHub/HF metadata limited to direct source-exposed paper-owned artifact URLs. Keep `tags`/`topic_hints` semantic and retrieval-friendly; never put URLs, arXiv/source IDs, DOI/source identifiers, route facts, resource/probe status, checksums, authors, categories, or API/link-health facts in them.

Raw-note body prose must stay source-grounded and source-clean: no bare URLs, Markdown links, route/probe/API/link-health facts, project/GitHub/HF/PWC details, repo/license/file-count/runtime audit details, resource-boundary paragraphs, or “link reachable but not audited” caveats. Keep exact routes and all resource/probe details in temp evidence, native-state reports, closeout reports, pending ledgers, or compact `log.md` entries.

## Integrated formula and mechanism evidence

Formula/objective evidence is mandatory reading evidence when present and belongs inside `## Methodology`, not in a standalone formula section. Preserve enough detail to reconstruct why the mechanism works: symbol roles, losses/objectives, routing rules, recurrence/update equations, normalization/scoring choices, protocol steps, or a clearly checked absence statement.

Use Obsidian-renderable TeX math syntax (`$...$` or `$$...$$`) plus explanatory words such as objective, loss, reward, score, normalization, or derivation when equations exist. Convert escaped MathJax delimiters `\(...\)` / `\[...\]` to dollar math before closeout. If there is no central equation, state the checked surfaces and use a formal substitute: manifest schema, taxonomy, severity rubric, protocol, algorithm stages, benchmark design, or state machine.

When writing formula-heavy notes, use a byte-safe composition path (`write_file`/`patch`, a Python raw string literal, or external body draft). After writing, scan the canonical file for control bytes and suspicious tabs; reopen the written file and confirm important TeX still has literal backslashes/braces.

## Integrated figure/table evidence

Figures, plots, architecture diagrams, screenshots, tables, panels, and qualitative examples are reading evidence, not raw-note artifacts. Do not embed images, copy figures into `raw/images/<slug>/`, paste chart/table links, store figure provenance blocks in the note, or reproduce full tables unless a tiny value subset is the shortest way to state the conclusion.

Write the conclusion beside the supported claim and include checkable anchors. For empirical plots/tables, include relevant values, row/column labels, metric names, axis coordinates, deltas, thresholds, comparison directions, or ablation trends. For conceptual figures/tables, include concrete labels/fields: axes, coordinates, rows/columns, channel/question text, named examples, or mapping pairs. Avoid chart-inventory prose (“Figure N has panels/axes/colors”) and broad claims (“Table N supports the framework”) without the data/fields that support them.

Place evidence by function: method/architecture diagrams -> `## Methodology`; benchmark tables/curves/ablations/panels -> `## 关键实验结果 / 作者结论`; failure-mode plots/tables -> `## 可能的局限`. Use verifier-visible labels such as `Figure 2`, `Table 1`, `图表`, `metric panel`, or `ablation curve` when relevant. If deriving percentages/ratios/means from table numbers, compute them with a tool and preserve compact source numbers/calculation in temp evidence or a durable `raw_fast_reports/*` sidecar before cleanup.

## Source and fallback evidence

For arXiv routes, resolve the canonical id/version and use API/abs/PDF/HTML/e-print/source as available. Treat arXiv HTML as semantic evidence only when the body actually contains the paper; a 200 page saying HTML is unavailable is not HTML evidence. For DOI/OJS/publisher/preprint routes blocked by bot challenges, use official metadata APIs, OAI/Crossref, supplied landing pages, and author-owned abstract/source surfaces as bounded fallback. State access limits in `## 可能的局限`; do not claim complete PDF/JATS or figure/table inspection when unavailable.

For long reports, surveys, dense appendices, source-heavy papers, or user requests to preserve subtle details, load `references/context-safe-long-source-clipping.md` and use scratch notes, thematic evidence batches, incremental draft, source reread, full-draft reread, and confidence review before the canonical raw-note write.

## Temporary evidence hygiene

Rendering figures, extracting tables, running visual QA, downloading PDFs/e-print source, and saving inventories are allowed only under `${LLM_WIKI_RAW_FAST_TMP_ROOT}/<slug>/...`. Copy only compact durable reports to `${LLM_WIKI_STATE_DIR}/raw_fast_reports/*` when needed before cleanup. Do not write these artifacts into the human wiki root, native workdir, or raw note.

For TeX/e-print sources, `${WIKI_GRAPH_REPO}/.agents/skills/llm-wiki/scripts/extract-tex-figures-tables.py` can enumerate figure/table environments. Treat output as an accelerator, not authority: cross-check important entries against source prose/PDF/HTML, run it over included section/table files when `main.tex` only has `\input{...}`, and exclude style/template/sample files before counting evidence.

## Resource boundary handoff

Resource facts are not note sections. `ops.raw_fast_ingest_prepare` performs source-exposed GitHub/HF/project/package triage and exact-link health by default; trust `agent_handoff.md` when `resource_review_required=false`. Do not inspect repo/HF substance, licenses, files, siblings, datasets, or project-page-discovered routes unless the user explicitly asks for a reproducibility/resource audit or the script returns `manual_reference_paths`. Treat unreachable exact links as unresolved/probe_failed, not verified absence; when no direct route exists, resource discovery is `not_checked` by default.

## Closeout and verification cues

After the note is written, use `references/raw-fast-closeout-ledger-contract.md` for exact verifier, closeout, pending wiki integration, and native freshness commands. Duplicate patterns are literal substrings, not regex; use source id, exact title, exact acronym, and a highly specific method phrase that actually appears in the note. Do not force URLs or resource facts into the raw note just to satisfy duplicate patterns.

A passing structured paper note has the required H2 sequence, section quality above the minimum floor, formula/objective evidence or checked absence in `## Methodology`, Obsidian-renderable math, figure/table-derived conclusions integrated into analytical prose, no deprecated standalone evidence headings, no Markdown image embeds, no copied figure/chart links, no resource/process sections, and no raw-note body probe/status leakage beyond allowed GitHub/HF artifact frontmatter fields.
