import sys
import argparse
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from support import sample_wiki, write  # noqa: E402
from ops import raw_fast_closeout  # noqa: E402
from ops.wiki_native_wiki_checks import wiki_root_machine_pollution  # noqa: E402
from ops.wiki_native_wiki_integration_pending import load_pending_wiki_integration_ledger  # noqa: E402

def _assert_timing_step(payload: dict, step: str) -> None:
    timings = payload["timings"]
    entry = timings["steps"][step]
    assert isinstance(entry["elapsed_seconds"], (int, float))
    assert entry["elapsed_seconds"] >= 0

def _structured_raw_fast_note(title: str, source: str) -> str:
    return f"""---
title: \"{title}\"
source: \"{source}\"
capture_route: \"test synthetic route\"
captured: \"2026-06-06 07:00 CST (+0800)\"
---

## 一句话总结

Synthetic take.

## 论文摘要（中文）

Synthetic abstract.

## Motivation

Synthetic motivation.

## Methodology

Formula evidence is integrated here: Eq. (1) defines $loss = x + y$ and the symbols are explained in the method narrative.

## 关键实验结果 / 作者结论

Figure 1 and Table 1 are integrated beside the result claim and record the observed trend.

## 对未来研究的启发

Future work can reuse the verification harness.

## 可能的局限

The tiny fixture is a synthetic limitation, not a real paper.

## 可继续追问的问题

Which wrapper gate catches failed verification before mark-pending?
"""
def _raw_fast_closeout_native_args(tmp_path: Path, mode: str) -> argparse.Namespace:
    return argparse.Namespace(
        root=tmp_path / "wiki",
        state_dir=tmp_path / "work" / "wikigraph" / "state",
        workdir=ROOT,
        timeout=17,
        refresh_timeout=23,
        native_refresh_mode=mode,
    )


def test_raw_fast_closeout_marks_pending_after_verifier_and_cleans_tmp(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "work" / "wikigraph" / "state"
    raw_rel = "raw/clip/2601/26010109_Tiny-Wrapper-Paper.md"
    title = "Tiny Wrapper Paper"
    source = "https://example.test/tiny-wrapper-paper.pdf"
    write(root / raw_rel, _structured_raw_fast_note(title, source))
    fetch_tmp = tmp_path / "fetch-tmp"
    write(fetch_tmp / "scratch.txt", "temporary evidence")
    write(
        fetch_tmp / "evidence_bundle.json",
        json.dumps(
            {
                "ok": True,
                "kind": "direct-pdf",
                "source_url": source,
                "title_guess": title,
                "warnings": [],
                "files": {"pdf": "paper.pdf", "paper_digest": "paper_digest.json", "localized_figures": "localized_figures.json"},
                "timings": {"total_seconds": 12.5, "steps": {"fetch_pdf": {"elapsed_seconds": 1.25}}},
            }
        ),
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ops.raw_fast_closeout",
            "--root",
            str(root),
            "--state-dir",
            str(state),
            "--workdir",
            str(ROOT),
            "--raw-file",
            raw_rel,
            "--title",
            title,
            "--source-id",
            source,
            "--pattern",
            title,
            "--pattern",
            source,
            "--topic-hint",
            "wrapper-test",
            "--resource-status-summary",
            "synthetic resources checked",
            "--tmp",
            str(fetch_tmp),
            "--auto-integrate",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)

    assert payload["raw_fast_ok"] is True
    assert payload["pre_verify"]["duplicate_strict_ok"] is True
    assert payload["control_scan"]["control_count"] == 0
    assert payload["marked"]["raw_path"] == raw_rel
    assert payload["wiki_integration"]["pending_count"] == 1
    assert payload["wiki_integration"]["should_integrate"] is False
    assert payload["native_refresh_status"]["blocked_by_pending_wiki_integration"] is True
    timing_steps = set(payload["timings"]["steps"])
    assert {"pre_verify", "mark_pending"} <= timing_steps
    assert payload["evidence_reports"]["count"] == 1
    evidence_report_path = Path(payload["evidence_reports"]["summaries"][0]["report_path"])
    assert evidence_report_path.exists()
    assert payload["final_verify"]["tmp_absent"][str(fetch_tmp)] is True
    assert not fetch_tmp.exists()
    ledger = load_pending_wiki_integration_ledger(state)
    assert ledger["pending"][0]["raw_path"] == raw_rel
    assert "resources" not in ledger["pending"][0]["required_sections"]


def test_raw_fast_closeout_compact_success_prints_session_summary(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "work" / "wikigraph" / "state"
    raw_rel = "raw/clip/2601/26010119_Compact-Wrapper-Paper.md"
    title = "Compact Wrapper Paper"
    source = "https://example.test/compact-wrapper-paper.pdf"
    write(root / raw_rel, _structured_raw_fast_note(title, source))

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ops.raw_fast_closeout",
            "--root",
            str(root),
            "--state-dir",
            str(state),
            "--workdir",
            str(ROOT),
            "--raw-file",
            raw_rel,
            "--title",
            title,
            "--source-id",
            source,
            "--pattern",
            source,
            "--output-mode",
            "compact",
        ],
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    visible_lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert 1 <= len(visible_lines) <= 8
    assert "raw_fast_ok=true" in result.stdout
    assert f"raw=`{raw_rel}`" in result.stdout
    assert "final_report=`" in result.stdout
    assert "wiki pending=1" in result.stdout
    for full_tree_key in ['"pre_verify":', '"evidence_reports":', '"timings":', '"final_verify":']:
        assert full_tree_key not in result.stdout


def test_raw_fast_closeout_compact_auto_integrate_keeps_runner_plan_and_native_summary_without_local_payload() -> None:
    compact = raw_fast_closeout.compact_auto_integrate(
        {
            "runner": "local",
            "plan_path": "/tmp/plan.json",
            "dry_run": False,
            "local_result": {
                "validation": {"input_fingerprints": {"raw": "bulk"}},
                "native_refresh": {
                    "native_refresh": True,
                    "skipped": False,
                    "runs": [
                        {
                            "refresh_kind": "incremental",
                            "build_executed": True,
                            "cutover_executed": True,
                            "active_already_fresh": False,
                            "pending_cleared": True,
                            "fill_missing_vectors": True,
                            "active": {"workspace_id": "candidate-after-wiki"},
                        }
                    ],
                    "status_after": {"pending_count": 0, "should_refresh": False},
                },
            },
            "post_status": {"pending_count": 0, "should_integrate": False},
        }
    )

    assert compact["runner"] == "local"
    assert compact["plan_path"] == "/tmp/plan.json"
    assert compact["post_status"]["pending_count"] == 0
    assert compact["native_refresh"]["run_count"] == 1
    assert compact["native_refresh"]["runs"][0]["build_executed"] is True
    assert compact["native_refresh"]["runs"][0]["active_already_fresh"] is False
    assert compact["native_refresh"]["runs"][0]["active_workspace_id"] == "candidate-after-wiki"
    assert compact["native_refresh"]["status_after"]["pending_count"] == 0
    assert "local_result" not in compact


def test_raw_fast_closeout_native_refresh_modes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cases = [
        ("status", 1, {"ran": False, "skipped": True, "skip_reason": "native_refresh_status_only"}),
        ("prepare", 2, {"ran": True, "prepared_only": True, "build_ok": True}),
    ]
    for mode, expected_call_count, expected_refresh in cases:
        args = _raw_fast_closeout_native_args(tmp_path, mode)
        status = {"command_returncode": 0, "pending_count": 1, "should_refresh": True}
        calls: list[dict[str, object]] = []

        def fake_run_json(command, *, cwd, timeout):
            calls.append({"command": command, "cwd": cwd, "timeout": timeout})
            if command[3] == "status":
                return {"returncode": 0, "json": {"pending_count": 1, "should_refresh": True}}
            assert mode == "prepare"
            assert command[1:5] == ["-m", "ops.batch_native_refresh", "refresh", "--prepare-only"]
            return {
                "returncode": 0,
                "json": {
                    "prepared_only": True,
                    "skipped": False,
                    "status_before": status,
                    "build": {"ok": True},
                },
            }

        monkeypatch.setattr(raw_fast_closeout, "run_json", fake_run_json)

        status_result = raw_fast_closeout.run_native_refresh_status(args)
        refresh_result = raw_fast_closeout.run_native_refresh_if_needed(args, status)

        assert len(calls) == expected_call_count
        status_command = calls[0]["command"]
        assert isinstance(status_command, list)
        assert status_command[1:4] == ["-m", "ops.batch_native_refresh", "status"]
        assert "--workdir" in status_command
        assert status_result["pending_count"] == 1
        assert status_result["command_returncode"] == 0
        assert refresh_result["refresh_mode"] == mode
        for key, value in expected_refresh.items():
            assert refresh_result[key] == value
        if mode == "status":
            assert refresh_result["pending_count"] == 1
        else:
            assert refresh_result["status"]["pending_count"] == 1


def test_raw_fast_closeout_does_not_mark_pending_when_verifier_fails(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "work" / "wikigraph" / "state"
    raw_rel = "raw/clip/2601/26010110_Bad-Wrapper-Paper.md"
    write(root / raw_rel, "---\ntitle: Bad\nsource: https://example.test/bad.pdf\n---\n\n## Methodology\n\nTODO\n")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ops.raw_fast_closeout",
            "--root",
            str(root),
            "--state-dir",
            str(state),
            "--workdir",
            str(ROOT),
            "--raw-file",
            raw_rel,
            "--title",
            "Bad Wrapper Paper",
            "--source-id",
            "https://example.test/bad.pdf",
            "--pattern",
            "https://example.test/bad.pdf",
        ],
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)

    assert result.returncode != 0
    assert payload["stage"] == "pre_verify"
    assert payload["raw_fast_ok"] is False
    pre_verify = payload["pre_verify"]
    assert pre_verify["verifier_path"].endswith("raw_fast_note_verify.py")
    assert pre_verify["diagnostic_hint"]["path"].endswith("raw_fast_note_verify.py")
    assert "repair anchor" in pre_verify["diagnostic_hint"]["message"]
    diagnostics = {item["code"]: item for item in pre_verify["blocker_diagnostics"]}
    assert "structured_evidence_sections_insufficient" in diagnostics
    assert "figure_table_evidence_integrated" in diagnostics
    assert diagnostics["figure_table_evidence_integrated"]["fix_hint"].startswith("Integrate figure/table")
    _assert_timing_step(payload, "pre_verify")
    assert "mark_pending" not in payload["timings"]["steps"]
    assert load_pending_wiki_integration_ledger(state)["pending"] == []


def test_raw_fast_closeout_compact_pre_verify_failure_prints_only_diagnostics(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "work" / "wikigraph" / "state"
    raw_rel = "raw/clip/2601/26010120_Bad-Compact-Wrapper-Paper.md"
    source = "https://example.test/bad-compact.pdf"
    write(root / raw_rel, f"---\ntitle: Bad Compact\nsource: {source}\n---\n\n## Methodology\n\nTODO\n")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ops.raw_fast_closeout",
            "--root",
            str(root),
            "--state-dir",
            str(state),
            "--workdir",
            str(ROOT),
            "--raw-file",
            raw_rel,
            "--title",
            "Bad Compact Wrapper Paper",
            "--source-id",
            source,
            "--pattern",
            source,
            "--output-mode",
            "compact",
        ],
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "raw_fast_ok: false" in result.stdout
    assert "stage: pre_verify" in result.stdout
    assert "structured_evidence_sections_insufficient" in result.stdout
    assert "figure_table_evidence_integrated" in result.stdout
    assert "repair anchor" in result.stdout
    for unrelated_key in ['"pre_verify":', '"timings":', '"mark_pending":', '"cleanup":', '"evidence_reports":']:
        assert unrelated_key not in result.stdout
    assert load_pending_wiki_integration_ledger(state)["pending"] == []


def test_raw_fast_closeout_refuses_non_tmp_cleanup_before_marking(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "work" / "wikigraph" / "state"
    raw_rel = "raw/clip/2601/26010111_Unsafe-Cleanup-Paper.md"
    title = "Unsafe Cleanup Paper"
    source = "https://example.test/unsafe-cleanup-paper.pdf"
    write(root / raw_rel, _structured_raw_fast_note(title, source))
    unsafe_tmp = root / "raw" / "clip" / "unsafe-bundle"
    write(unsafe_tmp / "scratch.txt", "must not be deleted")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ops.raw_fast_closeout",
            "--root",
            str(root),
            "--state-dir",
            str(state),
            "--workdir",
            str(ROOT),
            "--raw-file",
            raw_rel,
            "--title",
            title,
            "--source-id",
            source,
            "--pattern",
            source,
            "--tmp",
            str(unsafe_tmp),
            "--auto-integrate",
        ],
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)

    assert result.returncode != 0
    assert payload["stage"] == "cleanup_preflight"
    assert payload["cleanup_preflight"][0]["ok"] is False
    assert unsafe_tmp.exists()
    assert load_pending_wiki_integration_ledger(state)["pending"] == []


def test_raw_fast_closeout_compact_cleanup_failure_prints_only_failed_entry(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "work" / "wikigraph" / "state"
    raw_rel = "raw/clip/2601/26010121_Unsafe-Compact-Cleanup-Paper.md"
    title = "Unsafe Compact Cleanup Paper"
    source = "https://example.test/unsafe-compact-cleanup-paper.pdf"
    write(root / raw_rel, _structured_raw_fast_note(title, source))
    unsafe_tmp = root / "raw" / "clip" / "unsafe-compact-bundle"
    write(unsafe_tmp / "scratch.txt", "must not be deleted")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ops.raw_fast_closeout",
            "--root",
            str(root),
            "--state-dir",
            str(state),
            "--workdir",
            str(ROOT),
            "--raw-file",
            raw_rel,
            "--title",
            title,
            "--source-id",
            source,
            "--pattern",
            source,
            "--tmp",
            str(unsafe_tmp),
            "--output-mode",
            "compact",
        ],
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "raw_fast_ok: false" in result.stdout
    assert "stage: cleanup_preflight" in result.stdout
    assert str(unsafe_tmp) in result.stdout
    assert "refused_inside_wiki_root" in result.stdout
    for unrelated_key in ['"pre_verify":', '"control_scan":', '"timings":', '"mark_pending":']:
        assert unrelated_key not in result.stdout
    assert unsafe_tmp.exists()
    assert load_pending_wiki_integration_ledger(state)["pending"] == []


def test_raw_fast_closeout_final_verify_only_waives_post_integration_non_raw_hits() -> None:
    from ops import raw_fast_closeout

    pre = {"raw_fast_ok": True, "non_raw_wiki_hits": [], "raw_fast_blockers": []}
    final = {"raw_fast_ok": False, "non_raw_wiki_hits": ["concepts/after.md"], "raw_fast_blockers": ["non_raw_wiki_hits"]}
    assert raw_fast_closeout.final_verify_acceptable(pre, final) is True

    final_with_secret = dict(final, raw_fast_blockers=["non_raw_wiki_hits", "strict_secret_hits"])
    assert raw_fast_closeout.final_verify_acceptable(pre, final_with_secret) is False

    final_with_tmp_left = dict(final, tmp_absent={"/tmp/raw-fast": False})
    assert raw_fast_closeout.final_verify_acceptable(pre, final_with_tmp_left) is False


def test_raw_fast_closeout_fast_final_verify_records_tmp_absence(tmp_path: Path) -> None:
    state = tmp_path / "state"
    args = argparse.Namespace(state_dir=state, raw_file="raw/clip/2601/26010112_Fast-Final.md")
    tmp_bundle = tmp_path / "already-cleaned"
    pre = {
        "command_returncode": 0,
        "raw_fast_ok": True,
        "note_exists": True,
        "nonzero_size": True,
        "has_frontmatter": True,
        "frontmatter_fields_missing": [],
        "structured_sections_missing": [],
        "structured_evidence_sections_insufficient": [],
        "deprecated_standalone_evidence_sections": [],
        "structured_heading_order_ok": True,
        "duplicate_strict_ok": True,
        "non_raw_wiki_hits": [],
        "remote_markdown_images": 0,
        "data_uri_images": 0,
        "missing_local_images": 0,
        "strict_secret_hits": 0,
    }

    report = raw_fast_closeout.fast_final_verify_from_pre(args, pre, [tmp_bundle])

    assert report["raw_fast_ok"] is True
    assert report["fast_final_verify"] is True
    assert report["tmp_absent"] == {str(tmp_bundle): True}
    assert Path(report["report_path"]).exists()


def test_raw_fast_closeout_compact_log_contracts(tmp_path: Path) -> None:
    basic_args = argparse.Namespace(
        title="Compact Log Paper",
        source_id="https://arxiv.org/abs/2601.0101",
        raw_file="raw/clip/2601/26010112_Compact-Log-Paper.md",
        resource_status_summary="official abs/pdf/source verified; claimed code unresolved",
    )
    basic_output = {
        "raw_fast_ok": True,
        "final_verify": {"report_path": "/state/raw_fast_reports/compact_final_verify.json"},
        "wiki_integration": {"pending_count": 4, "actionable_pending_count": 4, "threshold": 10, "should_integrate": False, "next_required_action": "none"},
        "native_refresh_status": {"blocked_by_pending_wiki_integration": True, "graph_ready_pending_count": 0, "should_refresh": False},
    }
    basic_entry = raw_fast_closeout.build_compact_log_entry(basic_args, basic_output)

    assert len(basic_entry.splitlines()) <= 5
    assert "26010112_Compact-Log-Paper.md" in basic_entry
    assert "raw_fast_ok=true" in basic_entry
    assert "checksums" not in basic_entry.lower()

    blocked_args = argparse.Namespace(
        title="Blocked Native Ledger Paper",
        source_id="https://arxiv.org/abs/2601.0102",
        raw_file="raw/clip/2601/26010113_Blocked-Native-Ledger-Paper.md",
        resource_status_summary="official abs/pdf/source verified",
    )
    wiki_status = {
        "pending_count": 2,
        "actionable_pending_count": 2,
        "review_pending_count": 0,
        "blocking_pending_count": 2,
        "threshold": 10,
        "should_integrate": False,
        "next_required_action": "wiki_integration",
    }
    native_status = raw_fast_closeout.synthesize_blocked_native_refresh_status(
        blocked_args,
        wiki_status,
        standalone_status={
            "command_returncode": 0,
            "pending_count": 1,
            "should_refresh": True,
            "ledger_path": str(tmp_path / "state" / "pending_native_refresh.json"),
        },
    )
    blocked_entry = raw_fast_closeout.build_compact_log_entry(
        blocked_args,
        {
            "raw_fast_ok": True,
            "final_verify": {"report_path": "/state/raw_fast_reports/blocked_final_verify.json"},
            "wiki_integration": wiki_status,
            "native_refresh_status": raw_fast_closeout.compact_native_refresh_status(native_status),
        },
    )

    assert native_status["blocked_by_pending_wiki_integration"] is True
    assert native_status["graph_ready_pending_count"] == 0
    assert native_status["standalone_native_pending_count"] == 1
    assert native_status["standalone_native_should_refresh"] is True
    assert "graph-ready pending `0`" in blocked_entry
    assert "standalone native ledger pending `1`" in blocked_entry


def test_raw_fast_closeout_derives_args_from_evidence_bundle_without_mutation(tmp_path: Path) -> None:
    source_url = "https://arxiv.org/abs/2601.0101"
    tmp_bundle = tmp_path / "bundle"
    write(
        tmp_bundle / "evidence_report.json",
        json.dumps(
            {
                "resource_boundary": {
                    "status": "standardized",
                    "github_count": 0,
                    "hf_statuses": {"models": "not_checked"},
                }
            }
        ),
    )
    write(
        tmp_bundle / "evidence_bundle.json",
        json.dumps(
            {
                "ok": True,
                "kind": "arxiv",
                "source_url": source_url,
                "title_guess": "Closeout Derived Paper",
                "next_raw_path": "raw/clip/2601/26010114_Closeout-Derived-Paper.md",
                "preflight": {
                    "next_raw_path": "raw/clip/2601/26010114_Closeout-Derived-Paper.md",
                    "source_identity": {"arxiv_id_base": "2601.0101"},
                },
                "files": {"evidence_report": "evidence_report.json"},
            }
        ),
    )

    derived = raw_fast_closeout.derive_closeout_args_from_bundle(tmp_bundle / "evidence_bundle.json")

    assert derived["ok"] is True
    assert derived["raw_file"] == "raw/clip/2601/26010114_Closeout-Derived-Paper.md"
    assert derived["title"] == "Closeout Derived Paper"
    assert derived["source_id"] == source_url
    assert "Closeout Derived Paper" in derived["patterns"]
    assert source_url in derived["patterns"]
    assert "2601.0101" in derived["patterns"]
    assert derived["topic_hints"] == ["paper", "raw-fast", "arxiv"]
    assert "resource_boundary=standardized" in derived["resource_status_summary"]
    assert derived["tmp"] == [str(tmp_bundle)]
    assert "--raw-file" in derived["argv_tail"]
    assert not (tmp_bundle / "pending_wiki_integration.json").exists()


def test_raw_fast_closeout_captures_standardized_evidence_report(tmp_path: Path) -> None:
    tmp_bundle = tmp_path / "bundle"
    state = tmp_path / "state"
    write(
        tmp_bundle / "evidence_report.json",
        json.dumps(
            {
                "status": "standardized",
                "resource_boundary": {"status": "standardized", "github_count": 0, "hf_statuses": {"models": "not_checked"}},
                "paper_digest": {"status": "standardized", "equation_cards": 1},
            }
        ),
    )
    write(tmp_bundle / "agent_brief.json", json.dumps({"protected_anchors": {"next_raw_path": "raw/clip/2601/26010114_Report.md"}, "evidence_cards": [{"kind": "abstract"}]}))
    write(
        tmp_bundle / "evidence_bundle.json",
        json.dumps(
            {
                "ok": True,
                "kind": "arxiv",
                "source_url": "https://arxiv.org/abs/2601.0101",
                "title_guess": "Report Capture Paper",
                "files": {"evidence_report": "evidence_report.json", "agent_brief": "agent_brief.json"},
                "timings": {"total_seconds": 3.0},
            }
        ),
    )
    args = argparse.Namespace(tmp=[tmp_bundle], state_dir=state, raw_file="raw/clip/2601/26010114_Report.md")

    captured = raw_fast_closeout.capture_tmp_evidence_reports(args)

    assert captured["count"] == 1
    summary = captured["summaries"][0]
    assert summary["standardized_evidence_report"]["resource_boundary"]["status"] == "standardized"
    assert summary["agent_brief"]["evidence_card_count"] == 1
    report_path = Path(summary["report_path"])
    assert report_path.exists()
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved["standardized_evidence_report"]["paper_digest"]["status"] == "standardized"


def test_raw_fast_closeout_session_summary_is_compact_and_actionable(tmp_path: Path) -> None:
    final_report = {
        "ok": True,
        "raw_fast_ok": True,
        "raw_file": "raw/clip/2601/26010115_Summary-Paper.md",
        "final_verify": {
            "report_path": str(tmp_path / "state" / "raw_fast_reports" / "summary_final_verify.json"),
            "tmp_absent": {str(tmp_path / "bundle"): True},
        },
        "evidence_reports": {"count": 1, "summaries": [{"report_path": str(tmp_path / "state" / "raw_fast_reports" / "summary_evidence.json")} ]},
        "wiki_integration": {"pending_count": 5, "actionable_pending_count": 5, "threshold": 10, "should_integrate": False, "next_required_action": "none"},
        "native_refresh_status": {"blocked_by_pending_wiki_integration": True, "graph_ready_pending_count": 0, "standalone_native_pending_count": 1, "should_refresh": False},
        "compact_log_entry": "- 2026-01-01 raw-fast clip: `raw/clip/2601/26010115_Summary-Paper.md` — raw_fast_ok=true",
        "timings": {"total_seconds": 17.25},
    }

    summary = raw_fast_closeout.build_raw_fast_session_summary(final_report)

    assert summary["ok"] is True
    assert summary["raw_fast_ok"] is True
    assert summary["raw_file"] == final_report["raw_file"]
    assert summary["final_report"] == final_report["final_verify"]["report_path"]
    assert summary["wiki_pending"] == 5
    assert summary["native_blocked_by_wiki"] is True
    assert summary["tmp_absent_all"] is True
    assert "raw_fast_ok=true" in summary["markdown"]
    assert "wiki pending=5" in summary["markdown"]
    assert "stdout_tail" not in summary["markdown"]
    assert len(summary["markdown"].splitlines()) <= 8


def test_raw_fast_closeout_failure_summary_caps_stream_tail_and_omits_sibling_stages() -> None:
    output = {
        "ok": False,
        "stage": "native_refresh",
        "raw_fast_ok": False,
        "pre_verify": {"ok": True, "large_success_payload": "must stay hidden"},
        "native_refresh": {
            "ok": False,
            "returncode": 7,
            "stderr": "\n".join(f"stderr-line-{index:02d}" for index in range(25)),
        },
    }

    summary = raw_fast_closeout.build_raw_fast_failure_summary(output)
    markdown = summary["markdown"]

    assert summary["stage"] == "native_refresh"
    assert "returncode: 7" in markdown
    assert "pre_verify" not in markdown
    assert "stderr-line-04" not in markdown
    assert "stderr-line-05" in markdown
    assert "stderr-line-24" in markdown
    assert len([line for line in markdown.splitlines() if line.startswith("  stderr-line-")]) == 20
