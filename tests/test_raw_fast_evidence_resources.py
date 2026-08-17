from __future__ import annotations

import json
from pathlib import Path

import pytest

from raw_fast_evidence_fixtures import install_fake_arxiv_fetch, run_process_arxiv, sample_wiki


def test_raw_fast_evidence_bundle_resource_classifier_separates_hf_paper_index_collection_from_artifacts() -> None:
    from ops import raw_fast_evidence_bundle

    assert raw_fast_evidence_bundle._resource_url_candidate("https://huggingface.co/papers/2606.32039") is None
    assert raw_fast_evidence_bundle._resource_url_candidate("https://huggingface.co/models?other=arxiv:2606.32039") is None

    model = raw_fast_evidence_bundle._resource_url_candidate("https://huggingface.co/BinLin203/GEAR-VQ")
    dataset = raw_fast_evidence_bundle._resource_url_candidate("https://huggingface.co/datasets/BinLin203/GEAR-Data")
    collection = raw_fast_evidence_bundle._resource_url_candidate("https://huggingface.co/collections/BinLin203/gear-models")

    assert model is not None
    assert model["type"] == "hf"
    assert model["hf_kind"] == "models"
    assert model["repo"] == "BinLin203/GEAR-VQ"
    assert dataset is not None
    assert dataset["type"] == "hf"
    assert dataset["hf_kind"] == "datasets"
    assert dataset["repo"] == "BinLin203/GEAR-Data"
    assert collection is not None
    assert collection["type"] == "hf_collection"
    assert collection["hf_kind"] == "collections"
    assert collection["repo"] == "BinLin203/gear-models"

def test_raw_fast_evidence_bundle_pwc_supplied_resources_feed_metadata_without_hf_paper_confusion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from ops import raw_fast_evidence_bundle

    root = sample_wiki(tmp_path)
    workdir = tmp_path / "pwc-supplied-resource-metadata"
    supplied_url = "https://paperswithcode.co/paper/2606.32039"

    pwc_payload = {
        "arxiv_id": "2606.32039",
        "url_abs": "https://arxiv.org/abs/2606.32039",
        "url_pdf": "https://arxiv.org/pdf/2606.32039.pdf",
        "repositories": [
            {"url": "https://github.com/Tencent-Hunyuan/GEAR", "is_official": True, "source": "hf_api"},
        ],
        "project_pages": [
            {"url": "https://linb203.github.io/gear", "is_official": True, "source": "hf_api"},
        ],
        "hf_artifact_summary": {
            "best_kind": "model",
            "best_count": 2,
            "best_url": "https://huggingface.co/models?other=arxiv:2606.32039",
        },
        "urls_extracted": True,
    }
    hf_paper_payload = {
        "id": "2606.32039",
        "githubRepo": "https://github.com/Tencent-Hunyuan/GEAR",
        "projectPage": "https://linb203.github.io/gear",
        "linkedModels": [
            {"id": "BinLin203/Warmup-LFQ", "repoType": "model"},
            {"id": "BinLin203/GEAR-VQ", "repoType": "model"},
        ],
        "linkedDatasets": [
            {"id": "BinLin203/GEAR-Data", "repoType": "dataset"},
        ],
        "linkedSpaces": [
            {"id": "BinLin203/GEAR-Demo", "repoType": "space"},
        ],
    }
    hf_repos_payload = {
        "models": [
            {"id": "BinLin203/Warmup-LFQ"},
            {"id": "BinLin203/GEAR-VQ"},
        ],
        "datasets": [
            {"id": "BinLin203/GEAR-Data"},
        ],
        "spaces": [
            {"id": "BinLin203/GEAR-Demo"},
        ],
    }

    def fake_fetch_text(url: str, timeout: int) -> dict:
        if "export.arxiv.org" in url:
            return {"ok": True, "status": 200, "text": "<feed><entry><title>GEAR Fixture Paper</title></entry></feed>"}
        if url == "https://arxiv.org/abs/2606.32039":
            return {"ok": True, "status": 200, "text": "<html><title>GEAR Fixture Paper</title></html>"}
        if "paperswithcode.co/api/v1/papers/2606.32039" in url:
            return {"ok": True, "status": 200, "text": json.dumps(pwc_payload)}
        if url == "https://huggingface.co/api/papers/2606.32039":
            return {"ok": True, "status": 200, "text": json.dumps(hf_paper_payload)}
        if url == "https://huggingface.co/api/arxiv/2606.32039/repos":
            return {"ok": True, "status": 200, "text": json.dumps(hf_repos_payload)}
        raise AssertionError(f"unexpected fetch_text URL: {url}")

    install_fake_arxiv_fetch(
        monkeypatch,
        raw_fast_evidence_bundle,
        workdir,
        fetch_text=fake_fetch_text,
        extra=(
            (
                "probe_exact_link_health",
                lambda url, **kwargs: {"ok": True, "url": url, "status": "verified_present"},
            ),
        ),
    )

    payload = run_process_arxiv(
        raw_fast_evidence_bundle,
        supplied_url,
        root,
        workdir,
        probes=["arxiv", "doi"],
        timed=True,
    )

    assert payload["ok"] is True
    frontmatter = json.loads((workdir / "candidate_frontmatter.json").read_text(encoding="utf-8"))
    assert frontmatter["source"] == "https://arxiv.org/abs/2606.32039"
    assert frontmatter["github_links"] == ["https://github.com/Tencent-Hunyuan/GEAR"]
    assert frontmatter["huggingface_model_links"] == [
        "https://huggingface.co/BinLin203/GEAR-VQ",
        "https://huggingface.co/BinLin203/Warmup-LFQ",
    ]
    assert frontmatter["huggingface_dataset_links"] == ["https://huggingface.co/datasets/BinLin203/GEAR-Data"]
    metadata_text = json.dumps(frontmatter, sort_keys=True)
    assert "https://huggingface.co/papers/2606.32039" not in metadata_text
    assert "https://huggingface.co/models?other=arxiv:2606.32039" not in metadata_text
    assert "https://huggingface.co/spaces/BinLin203/GEAR-Demo" not in metadata_text
    supplied_resources = json.loads((workdir / "supplied_page_resources.json").read_text(encoding="utf-8"))
    assert supplied_resources["ok"] is True
    assert set(supplied_resources["platforms_checked"]) == {"paperswithcode", "huggingface"}

def test_raw_fast_evidence_bundle_modelscope_supplied_resources_feed_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from ops import raw_fast_evidence_bundle

    root = sample_wiki(tmp_path)
    workdir = tmp_path / "modelscope-supplied-resource-metadata"
    supplied_url = "https://modelscope.ai/papers/2606.07591"
    modelscope_html = r'''
    <html><head><title>ResearchClawBench</title></head><body>
    <script id=\"__MODEL_SCOPE_DATA__\">
    {\\\"ArxivId\\\":\\\"2606.07591\\\",\\\"ArxivUrl\\\":\\\"https://arxiv.org/abs/2606.07591\\\",\\\"CodeLink\\\":\\\"https://github.com/InternScience/ResearchClawBench\\\"}
    </script>
    <meta property=\"og:image\" content=\"https://cdn.modelscope.ai/social-thumbnails/papers/2606.07591.png\">
    </body></html>
    '''

    def fake_fetch_text(url: str, timeout: int) -> dict:
        if "export.arxiv.org" in url:
            return {"ok": True, "status": 200, "text": "<feed><entry><title>ResearchClawBench Fixture Paper</title></entry></feed>"}
        if url == "https://arxiv.org/abs/2606.07591":
            return {"ok": True, "status": 200, "text": "<html><title>ResearchClawBench Fixture Paper</title></html>"}
        if url == supplied_url:
            return {"ok": True, "status": 200, "text": modelscope_html}
        raise AssertionError(f"unexpected fetch_text URL: {url}")

    install_fake_arxiv_fetch(
        monkeypatch,
        raw_fast_evidence_bundle,
        workdir,
        fetch_text=fake_fetch_text,
        extra=(
            (
                "probe_exact_link_health",
                lambda url, **kwargs: {"ok": True, "url": url, "status": "verified_present"},
            ),
        ),
    )

    payload = run_process_arxiv(
        raw_fast_evidence_bundle,
        supplied_url,
        root,
        workdir,
        probes=["arxiv", "doi"],
        timed=True,
    )

    assert payload["ok"] is True
    assert payload["source_url"] == "https://arxiv.org/abs/2606.07591"
    assert payload["supplied_url"] == supplied_url
    frontmatter = json.loads((workdir / "candidate_frontmatter.json").read_text(encoding="utf-8"))
    assert frontmatter["source"] == "https://arxiv.org/abs/2606.07591"
    assert frontmatter["github_links"] == ["https://github.com/InternScience/ResearchClawBench"]
    supplied_resources = json.loads((workdir / "supplied_page_resources.json").read_text(encoding="utf-8"))
    assert supplied_resources["ok"] is True
    assert supplied_resources["platforms_checked"] == ["modelscope"]
    assert supplied_resources["candidate_count"] == 1
    assert supplied_resources["candidates"][0]["source"] == "modelscope_page.CodeLink"
    assert "cdn.modelscope.ai" not in json.dumps(supplied_resources, sort_keys=True)

def test_raw_fast_evidence_bundle_only_verified_abstract_source_links_enter_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    from ops import raw_fast_evidence_bundle

    def fake_probe(url: str, **kwargs) -> dict:
        ok = "model-bad" not in url
        return {"ok": ok, "url": url, "status": "verified_present" if ok else "probe_failed"}

    monkeypatch.setattr(
        raw_fast_evidence_bundle,
        "probe_exact_link_health",
        fake_probe,
    )

    probe = raw_fast_evidence_bundle.build_resource_probe(
        r"""
        \begin{abstract}Official implementation: https://github.com/example/paper-code,
        model weights: https://huggingface.co/example/paper-model, dataset:
        https://huggingface.co/datasets/example/paper-data, demo:
        https://huggingface.co/spaces/example/paper-demo, and unresolved model:
        https://huggingface.co/example/model-bad before closing.
        \end{abstract}
        % Optional math commands from https://github.com/goodfeli/dlbook_notation.
        \\input{math_commands.tex}
        Related work uses https://github.com/example/reference-code and https://huggingface.co/example/reference-model.
        """,
        {"links": []},
        [],
        health_mode="direct",
    )

    links = raw_fast_evidence_bundle.metadata_resource_links(probe)
    assert links == {
        "github_links": ["https://github.com/example/paper-code"],
        "huggingface_model_links": ["https://huggingface.co/example/paper-model"],
        "huggingface_dataset_links": ["https://huggingface.co/datasets/example/paper-data"],
    }
    source_exposed = probe["source_exposed_resources"]
    assert [item["url"] for item in source_exposed["github"]] == ["https://github.com/example/paper-code"]
    hf_by_url = {item["url"]: item for item in source_exposed["hf"]}
    assert hf_by_url["https://huggingface.co/example/paper-model"]["status"] == "verified_present"
    assert hf_by_url["https://huggingface.co/datasets/example/paper-data"]["status"] == "verified_present"
    assert hf_by_url["https://huggingface.co/spaces/example/paper-demo"]["hf_kind"] == "spaces"
    assert hf_by_url["https://huggingface.co/example/model-bad"]["status"] == "probe_failed"
    ignored_reasons = {item["url"]: item["reason"] for item in source_exposed["ignored"]}
    assert ignored_reasons["https://github.com/goodfeli/dlbook_notation"] == "auxiliary_tex_notation_or_template_repo"
    assert ignored_reasons["https://github.com/example/reference-code"] == "non_abstract_source_resource_link"
    assert ignored_reasons["https://huggingface.co/example/reference-model"] == "non_abstract_source_resource_link"

def test_raw_fast_evidence_bundle_resource_boundary_defaults_to_not_checked_without_exact_link_report() -> None:
    from ops import raw_fast_evidence_bundle

    resource_probe = {
        "ok": True,
        "probes": [
            {"ok": True, "type": "doi", "doi": "10.1234/example.paper", "status": "detected", "url": "https://doi.org/10.1234/example.paper"},
            {"ok": True, "type": "arxiv", "id": "2604.08999", "status": "detected", "url": "https://arxiv.org/abs/2604.08999"},
            {"ok": True, "type": "github_repo", "repo": "example/ignored"},
            {"ok": True, "type": "hf_models", "query": "Fixture", "count": 0, "items": []},
            {"ok": True, "type": "project_page", "url": "https://example.test/project"},
        ],
    }

    draft = raw_fast_evidence_bundle.summarize_resource_boundary(resource_probe, metadata={"title": "Fixture Sidecar Paper"})
    summary = raw_fast_evidence_bundle.resource_status_summary(draft)

    assert draft["github"] == []
    assert draft["project_pages"] == []
    assert {kind: bucket["status"] for kind, bucket in draft["hf"].items()} == {"models": "not_checked", "datasets": "not_checked", "spaces": "not_checked"}
    assert draft["doi"][0]["doi"] == "10.1234/example.paper"
    assert draft["arxiv"][0]["id"] == "2604.08999"
    assert "verified_absent" not in summary
    assert "candidates_unverified" not in summary
    assert "review_required=no" in summary

def test_raw_fast_evidence_bundle_frontmatter_domain_contracts() -> None:
    from ops import raw_fast_evidence_bundle

    fm = raw_fast_evidence_bundle.build_frontmatter(
        "Compact Candidate Paper",
        "https://arxiv.org/abs/2601.0101",
        "arxiv",
    )

    assert fm["domain"] == "machine-learning"
    assert fm["domain"] != "paper"
    assert set(fm) <= {
        "title",
        "source",
        "created",
        "updated",
        "type",
        "domain",
        "tags",
        "topic_hints",
        "capture_route",
        "captured",
    }
    assert "resource_status" not in fm
    assert "source_id" not in fm

    skeleton = raw_fast_evidence_bundle.build_note_skeleton(fm)
    assert "## 资源与复现状态" not in skeleton
    assert "## Evidence trail" not in skeleton
    expected_domains = {
        ("cs.CV", "cs.LG"): "computer-vision",
        ("cs.LG", "stat.ML"): "machine-learning",
        ("cs.RO",): "robotics",
        (): "machine-learning",
    }
    for categories, expected in expected_domains.items():
        assert raw_fast_evidence_bundle.domain_from_arxiv_categories(list(categories)) == expected
    assert raw_fast_evidence_bundle.domain_from_arxiv_categories(["cs.LG"]) != "paper"

def test_raw_fast_evidence_bundle_resource_probe_contracts_are_explicit() -> None:
    from ops import raw_fast_evidence_bundle

    detected = raw_fast_evidence_bundle.build_resource_probe(
        "See arXiv:2604.08999 and DOI 10.1234/example.paper for details.",
        {"links": [{"uri": "https://github.com/example/sidecar"}]},
        ["arxiv", "doi"],
    )
    probes = {(item["type"], item.get("id") or item.get("doi")): item for item in detected["probes"]}

    assert ("arxiv", "2604.08999") in probes
    assert probes[("arxiv", "2604.08999")]["ok"] is True
    assert probes[("arxiv", "2604.08999")]["status"] == "detected"
    assert probes[("arxiv", "2604.08999")]["evidence"]
    assert ("doi", "10.1234/example.paper") in probes
    assert probes[("doi", "10.1234/example.paper")]["status"] == "detected"
    assert "https://github.com/example/sidecar" in detected["urls"]

    skipped = raw_fast_evidence_bundle.build_resource_probe(
        "Project page: https://example.test/project and arXiv:2604.08999.",
        {"links": [{"uri": "https://huggingface.co/example/model"}]},
        ["none"],
    )

    assert skipped["ok"] is True
    assert skipped["skipped"] is True
    assert skipped["probes"] == []
    assert skipped["urls"] == ["https://example.test/project", "https://huggingface.co/example/model"]
