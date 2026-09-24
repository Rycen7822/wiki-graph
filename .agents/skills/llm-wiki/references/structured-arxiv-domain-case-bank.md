# Structured arXiv domain case bank

Use after `references/structured-paper-ingest-router.md` selects a domain-shape companion. This file owns domain-specific evidence preservation; generic note shape, discovery, closeout, ledgers, and native refresh stay elsewhere.

Load separately: `references/structured-paper-source-route-cases.md` for source-exposed route identity, `references/context-safe-long-source-clipping.md` for dense sources, and `references/raw-fast-closeout-ledger-contract.md` for closeout. Resource exact-link policy is automated by `ops.raw_fast_ingest_prepare`; load manual resource docs only from script-returned `manual_reference_paths` or explicit audit scope.

## Shared boundaries

- Default clipping verifies only exact source-exposed routes; no GitHub/HF search by title, acronym, arXiv ID, method, organization, author, benchmark, or project page unless the user asks for an audit.
- Raw-note frontmatter/body stay source-clean: no resource-status fields, Markdown links, route/link-health/API/repo/license/file-count/runtime/audit facts, or standalone resource/process headings. The only default resource URL fields are direct source-exposed paper-owned `github_links`, `huggingface_model_links`, and `huggingface_dataset_links`.
- Put formulas/objectives/theorems/protocols in `## Methodology`; table/figure-derived conclusions beside supported claims; no standalone formula/figure sections, images, or wholesale tables.
- Audit details live in tmp evidence, raw-fast reports, closeout, ledgers, or compact log. Use context-safe scratch for appendix-heavy, source-bundle-heavy, formula/table dense sources.

## Agentic systems and AI R&D automation

Trigger: self-improving agents, AI-scientist loops, tool-use protocols, memory/state, search/evolution, deployment-time adaptation, lab-in-the-loop discovery, SWE-agent/repository issue solving, delegation swarms, or agents doing training/post-training/data curation/benchmark optimization/ML engineering.

Preserve layers: paper evidence from arXiv/PDF/source tables, figures, formulas, traces, prompts, settings, protocols; current repo evidence only in audit mode; dependencies for base models/APIs/tools/datasets/instruments/human queues; reproduction boundary for hidden services, private traces, unverified model substitution, hidden tests, private judges, or later framework claims.

Methodology anchors: policy/state formalism, tool set, memory/store, allowed actions, update/rollback, solve-vs-evolve, verifier/test/reward gate, acceptance threshold, provenance, invalid-submission rule, search/history schedule, budget controls, process metrics. For AI R&D benchmarks, preserve instance definition, objective, compute/time budget, allowed/disallowed actions, internet/data access, final artifact, scoring/aggregation, violation fallback, and judge/audit protocol. Results/limitations should bind leaderboards/traces to narrow-task wins, token/time tradeoffs, violations, contamination, model substitution, unauthorized services, harness modification, context-loss failures, and concrete failure modes. Do not conflate README-current claims with paper experiments, framework repos with full reproduction packages, HF paper pages with releases, replay traces with datasets, or dependency links with paper-owned artifacts.

## Benchmark, dataset, and evaluation harness releases

Trigger: paper directly links a benchmark, dataset, evaluation harness, generated task set, simulator, leaderboard, video/state data, geometry/spatial dataset, anonymous review repo, or project data route, especially when artifact substance is requested.

Default stops at direct link health. Audit mode resolves paper identity, then inspects project/GitHub/HF/dataset-server/anonymous-review routes. HF checks: public/gated/disabled state, card license/tags/siblings/splits, rows/files/bytes/hash, parser failures, collision risk. GitHub checks: API/root tree/README/license/default branch plus prompts, metrics, configs, tasks, data, specs, runners, simulators, logs. Compare advertised scale with public artifact scale and record partial-kit mismatches.

Raw-note method describes benchmark construction: sampling/filtering, label derivation, prompt protocol, metrics, contamination controls, audit thresholds, item set size, judge/scoring, aggregation, capability map, or version/update protocol. Preserve taxonomies, answer-form splits, capability decompositions, human-reference protocols, and pipelines. Compute key table deltas with tools; call a harness “full reproduction” only when raw data, scripts, model calls, logs, and environment are sufficient.

## Latent looped / fixed-point reasoning models

Trigger: looped Transformers, recurrent-depth latent reasoning, fixed-point/equilibrium reasoning, adaptive compute, ACT replacement, HRM/TRM-style symbolic reasoning, signal-propagation analyses.

Preserve recurrence, state variable, stopping rule, stability/boundedness, contraction/convergence evidence, and whether accuracy saturation differs from fixed-point convergence. For ACT/halting, record inference-time ACT usage, batching until last halt, and compute proxy. Treat hierarchy claims as signal-propagation/optimization hypotheses unless proven. Read tables by model size/task; pair accuracy and compute; distinguish budget exhaustion, early halt failure, and Pareto improvements. Topic hints: looped transformer adaptive compute, fixed-point halting, pre-norm residual scaling, state tracking length generalization, Sudoku/Maze/ARC-AGI, damped fixed-point optimizer, truncated BPTT, hierarchy versus propagation.

## Mechanistic theory with code/demo routes

Trigger: dense mechanistic interpretability, grokking, algorithmic-task theory, neural-network theory, training dynamics, Fourier/spectral features, ODE/gradient-flow dynamics, lottery-ticket emergence, synthetic tasks, or theorem-backed mechanisms with direct GitHub/HF Space/project/demo/visualization routes.

Use long-source scratch when dense. Read batches: problem/model/objective and spectral setup; theorem/proof skeletons; ODE/gradient-flow derivations; experiments/ablations/grokking figures and appendix caveats; exact source-exposed routes. For source bundles, run figure/table extraction over included section files, not only a top-level `main.tex` stub.

Methodology anchors: task/model/loss definitions, DFT/spectral coordinates, theorem/proposition statements, ODEs, convergence-time claims, proof assumptions. Results anchors: DFT heatmaps, phase plots, activation-swapping tables, ablation tables, lottery/grokking curves, appendix heatmaps. Put sign inconsistencies, min/max misalignment, impossible cancellation, label/caption tension, or theorem/proof/caption conflicts in limitations when they affect reuse. Exact code/demo routes get link-health by default, not repo/tree/license/demo-backend audit.

## Theory, formal analysis, and proof-result/data releases

Trigger: theorem/proof/algorithm analysis, controlled synthetic experiments, sample complexity, latent-prediction/JEPA/data2vec theory, tokenizer/byte-simulation scaling, steering-vector distillation, mechanistic RL/post-training analysis, Lean/Coq/Isabelle/Mathlib proof search, compiler/verifier feedback, AND-OR trees/DAGs, theorem-prover benchmarks, olympiad formalization, proof-result files, theorem CSVs, leaderboards, or project pages where full runtime may be absent.

Theory anchors: assumptions, definitions, theorem statements, ODEs/bounds, limiting distributions, proof skeletons, pseudocode, tokenizer compression or bits-per-byte objectives, steering-vector projections, perturbation/Taylor terms, subspace projections, contraction inequalities, proxy scores, caveats. Formal-prover anchors: proof goals, subgoal decomposition, OR/AND tree or DAG semantics, compiler/kernel feedback, reviewer/critic filters, memoization, backtracking, and parent-proof certification.

Artifact layers: proof-result/data release can mean first-party theorem statements, benchmark CSVs, proof files, leaderboards, PDFs, or static project pages; partial reproducibility means inspect/check artifacts when dependencies are available; full-system reproduction requires public runtime, prompts, compiler-feedback loop, state graph, reviewer/critic, search policy, rollouts, retrieval pieces, trajectories/logs, scoring scripts, and model/provider access. Keep artifact counts, CSV headers, branch/license/subdirectory facts, HF results, and route health external unless interpretation-relevant.

## Video world models with latent or 3D memory

Trigger: video/world-generation with persistent spatial memory, 3D caches, RGB point clouds, voxel/grid/latent memory, novel-view synthesis, camera trajectories, closed-loop revisit evaluation, or mismatch between external 3D cache and latent diffusion/video backbone.

Methodology anchors: cache representation and read/update cycle; RGB baseline `Rasterise(cache) -> VAE encoder -> latent`; latent-memory tuple `(world point, latent feature/token)`; depth-guided back-projection, target-view projection, z-buffer readout, visibility masks, dynamic-object filtering, video-model injection, scaled intrinsics, pinhole back-projection, candidate cells, update masks, chunking, VAE stride/channel count, ControlNet/LoRA/side branch, training stages, rollout algorithm. Results anchors: WorldScore/novel-view/closed-loop metrics, revisit/open-domain figures, efficiency scaling, ablation rows; compute small deltas with tools. Limitations: static-geometry caches often exclude moving objects or sky, so actor state may not persist. Project page/shortlink is not code/checkpoint/demo release; without direct GitHub/HF artifact URL, do not search by title/acronym/author/project page.

## Verification reminders

Use exact duplicate patterns: arXiv ID, exact title, DOI, and paper-owned resource URL only when intentionally present in a schema-compliant surface. Do not use dependencies, base models, theorem-prover names, benchmark suite names, short project pages, or family repos as strict duplicate patterns. After marking pending, report raw saved/verified, wiki integration pending, threshold count, and graph blocker state.