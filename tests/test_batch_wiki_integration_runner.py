import sys
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

from support import append_native_history as _append_native_history  # noqa: E402
from support import mark_pending_batch, sample_wiki, write  # noqa: E402
from ops import batch_native_refresh  # noqa: E402
from ops import batch_wiki_integration  # noqa: E402
from ops.wiki_native_wiki_integration_pending import DEFAULT_PENDING_WIKI_INTEGRATION_THRESHOLD  # noqa: E402
from ops.wiki_native_wiki_integration_pending import load_pending_wiki_integration_ledger  # noqa: E402


def _fake_native_refresh(native_calls):
    def fake(root_arg, state_arg, *, workdir, reason, defer_native_refresh=False):
        native_calls.append((root_arg, state_arg, workdir, reason, len(batch_native_refresh.pending_entries(state_arg))))
        batch_native_refresh.clear_pending(state_arg)
        return 0, {
            "native_refresh": True,
            "runs": [
                {
                    "refresh_kind": "incremental",
                    "fill_missing_vectors": True,
                    "vector_cache_required": True,
                }
            ],
            "status_after": batch_native_refresh.status(root_arg, state_arg),
        }

    return fake


def _write_fake_integrator(path: Path, *, read_prompt: bool = False) -> None:
    extra = (
        "seen = state / 'fake_runner_seen.txt'\n"
        "seen.write_text(prompt_path.read_text(encoding='utf-8')[:1600], encoding='utf-8')\n"
        if read_prompt
        else ""
    )
    write(
        path,
        "import os, sys\n"
        "from pathlib import Path\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "from ops.wiki_native_wiki_integration_bridge import clear_pending_wiki_integration_after_success\n"
        "root = Path(os.environ['LLM_WIKI_ROOT'])\n"
        "state = Path(os.environ['LLM_WIKI_STATE_DIR'])\n"
        "prompt_path = Path(os.environ['LLM_WIKI_INTEGRATION_PROMPT'])\n"
        f"{extra}"
        "clear_pending_wiki_integration_after_success(root, state, reason=os.environ.get('LLM_WIKI_INTEGRATION_REASON', 'threshold'))\n",
    )


def _assert_native_deferred(payload: dict, state: Path, *, nested: bool = False) -> None:
    body = payload["auto_integrate"] if nested else payload
    native = body.get("native_refresh") or body["local_result"]["native_refresh"]
    assert native["deferred"] is True
    if "skipped" in native:
        assert native["skipped"] is True
    if "status_after" in native:
        assert native["status_after"]["should_refresh"] is True
    assert body["post_status"]["pending_count"] == 0
    assert [entry["reason"] for entry in batch_native_refresh.pending_entries(state)] == ["wiki-integration:threshold"]


def test_batch_wiki_integration_auto_integrate_defaults_to_local_runner_at_threshold(tmp_path: Path, monkeypatch) -> None:
    root = sample_wiki(tmp_path)
    write(
        root / "index.md",
        "# LLM Wiki Index\n\n> Last updated: 2026-05-18 16:00 | Total pages: 4\n\n"
        "## Concepts\n\n- [[foo]] - Foo page.\n\n"
        "## Queries\n\n- [[bar]] - Bar page.\n\n"
        "## Meta\n\n- [[raw-clip-map]] - Raw clip map.\n- [[topic-map]] - Topic map.\n",
    )
    state = tmp_path / "work" / "wikigraph" / "state"
    mark_pending_batch(state, root, 10, topic_hints=["local runner topic"], required_sections=["summary"])

    native_calls = []
    monkeypatch.setattr(
        batch_wiki_integration,
        "run_native_refresh_after_wiki_integration",
        _fake_native_refresh(native_calls),
    )

    code, payload = batch_wiki_integration.run_auto_integration(root, state, reason="threshold")

    assert code == 0
    assert payload["ran"] is True
    assert payload["runner"] == "local"
    assert payload["command"][0] == "integrate-local"
    assert payload["local_result"]["apply"]["operations_applied"] >= 11
    assert payload["local_result"]["validation"]["errors_count"] == 0
    assert payload["local_result"]["native_refresh"]["runs"][0]["fill_missing_vectors"] is True
    assert payload["post_status"]["pending_count"] == 0
    assert load_pending_wiki_integration_ledger(state)["pending"] == []
    assert len(batch_native_refresh.pending_entries(state)) == 0
    assert native_calls == [(root, state, ROOT, "threshold", 1)]


def test_batch_wiki_integration_auto_integrate_runs_configured_runner_at_threshold_and_requires_cleared_ledger(tmp_path: Path, monkeypatch) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "work" / "wikigraph" / "state"
    mark_pending_batch(state, root, 10, path_prefix="260106", required_sections=["summary"])
    fake_runner = tmp_path / "fake_wiki_integrator.py"
    _write_fake_integrator(fake_runner, read_prompt=True)
    native_calls = []
    monkeypatch.setattr(
        batch_wiki_integration,
        "run_native_refresh_after_wiki_integration",
        _fake_native_refresh(native_calls),
    )

    code, payload = batch_wiki_integration.run_auto_integration(
        root,
        state,
        reason="threshold",
        integration_command=f"{sys.executable} {fake_runner}",
    )

    assert code == 0
    assert payload["ran"] is True
    assert payload["pre_status"]["should_integrate"] is True
    assert payload["post_status"]["should_integrate"] is False
    assert payload["post_status"]["pending_count"] == 0
    assert payload["native_refresh"]["runs"][0]["fill_missing_vectors"] is True
    assert payload["prompt_path"].endswith(".md")
    assert payload["plan_operations"] >= 1
    assert Path(payload["plan_path"]).exists()
    assert "Plan artifact" in (state / "fake_runner_seen.txt").read_text(encoding="utf-8")
    assert load_pending_wiki_integration_ledger(state)["pending"] == []
    assert len(batch_native_refresh.pending_entries(state)) == 0
    assert native_calls == [(root, state, ROOT, "threshold", 1)]


def _set_native_cutover_env(monkeypatch, watched_dir: Path) -> None:
    monkeypatch.setenv("LLM_WIKI_NATIVE_RESTART_COMMAND", f"{sys.executable} -c pass")
    monkeypatch.setenv("LLM_WIKI_NATIVE_SMOKE_URL", "http://127.0.0.1:9621/query/data")
    monkeypatch.setenv("LLM_WIKI_NATIVE_SMOKE_QUERY", "native refresh smoke")
    monkeypatch.setenv("LLM_WIKI_NATIVE_UNCHANGED_PATH", str(watched_dir))


def _stub_semantic_artifact_refresh(monkeypatch) -> None:
    monkeypatch.setattr(
        batch_wiki_integration.native_semantic_artifact_refresh,
        "refresh_semantic_artifacts",
        lambda *args, **kwargs: {"ok": True, "integrated_paths": ["raw/clip/2601/26010101_Foo-Paper.md"], "failures": []},
    )
    monkeypatch.setattr(
        batch_wiki_integration.native_semantic_artifact_refresh,
        "validate_active_workspace_coverage",
        lambda *args, **kwargs: {"ok": True, "workspace_id": "active-test", "covered_path_count": 1, "failures": []},
    )


def test_native_refresh_followthrough_runs_incremental_then_due_full_rebuild_with_vector_cache(tmp_path: Path, monkeypatch) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "work" / "wikigraph" / "state"
    watched_dir = tmp_path / "watched"
    watched_dir.mkdir()
    _append_native_history(state, [f"native graph incremental refresh: cutover {idx}" for idx in range(4)])
    batch_native_refresh.mark_pending(state, root, reason="wiki-integration:threshold")
    _set_native_cutover_env(monkeypatch, watched_dir)
    _stub_semantic_artifact_refresh(monkeypatch)
    runs = []
    real_refresh_cutover = batch_native_refresh.refresh_cutover

    def recording_refresh_cutover(**kwargs):
        runs.append(
            (
                batch_native_refresh.status(kwargs["root"], kwargs["state_dir"])["next_refresh_kind"],
                kwargs["workspace_id"],
                kwargs["fill_missing_vectors"],
                kwargs["force"],
            )
        )
        assert kwargs["force"] is True

        def build_workspace(**build_kwargs):
            return {"ok": True, "workspace_id": build_kwargs["workspace_id"]}

        def finalize_workspace(*, state_dir, reason):
            history = batch_native_refresh.active_workspace_history_path(state_dir)
            history.parent.mkdir(parents=True, exist_ok=True)
            with history.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps({"reason": reason, "current": {"workspace_id": kwargs["workspace_id"]}}) + "\n"
                )
            return {"schema_version": 1, "workspace_id": kwargs["workspace_id"], "status": "active"}

        def restart_service(*, state_dir):
            return {"service": "llm-wiki-native", "status": "ok"}

        def query_smoke(*, state_dir, active):
            return {"ok": True, "url": "http://127.0.0.1:9621/query/data"}

        return real_refresh_cutover(
            **{
                **kwargs,
                "build_workspace": build_workspace,
                "finalize_workspace": finalize_workspace,
                "restart_service": restart_service,
                "query_smoke": query_smoke,
            }
        )

    monkeypatch.setattr(batch_wiki_integration.batch_native_refresh, "refresh_cutover", recording_refresh_cutover)

    code, payload = batch_wiki_integration.run_native_refresh_after_wiki_integration(root, state, workdir=ROOT, reason="threshold")

    assert code == 0
    assert [row[0] for row in runs] == ["incremental", "full-rebuild"]
    assert all(row[2] is True for row in runs)
    assert payload["status_after"]["should_refresh"] is False
    assert payload["semantic_artifacts"]["ok"] is True
    assert payload["active_workspace_coverage"]["ok"] is True
    assert batch_native_refresh.pending_entries(state) == []


def test_native_refresh_followthrough_surfaces_cutover_failure_without_clearing_pending(tmp_path: Path, monkeypatch) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "work" / "wikigraph" / "state"
    watched_dir = tmp_path / "watched"
    watched_dir.mkdir()
    batch_native_refresh.mark_pending(state, root, reason="wiki-integration:threshold")
    _set_native_cutover_env(monkeypatch, watched_dir)
    _stub_semantic_artifact_refresh(monkeypatch)

    def fake_refresh_cutover(**kwargs):
        raise RuntimeError("native service smoke failed")

    monkeypatch.setattr(batch_wiki_integration.batch_native_refresh, "refresh_cutover", fake_refresh_cutover)

    code, payload = batch_wiki_integration.run_native_refresh_after_wiki_integration(root, state, workdir=ROOT, reason="threshold")

    assert code == 16
    assert payload["failure"]["reason"] == "native-refresh-failed"
    assert "native service smoke failed" in payload["failure"]["message"]
    assert payload["status_after"]["should_refresh"] is True
    assert len(batch_native_refresh.pending_entries(state)) == 1


def test_native_refresh_stops_before_cutover_when_semantic_artifact_gate_fails(tmp_path: Path, monkeypatch) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "work" / "wikigraph" / "state"
    batch_native_refresh.mark_pending(state, root, reason="wiki-integration:threshold")
    report = {"ok": False, "failures": [{"code": "raw-section-stale"}]}

    def fail_semantic_refresh(*args, **kwargs):
        raise batch_wiki_integration.native_semantic_artifact_refresh.SemanticArtifactRefreshError(
            "semantic gate failed",
            report=report,
        )

    monkeypatch.setattr(
        batch_wiki_integration.native_semantic_artifact_refresh,
        "refresh_semantic_artifacts",
        fail_semantic_refresh,
    )
    monkeypatch.setattr(
        batch_wiki_integration.batch_native_refresh,
        "refresh_cutover",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("cutover must not run")),
    )

    code, payload = batch_wiki_integration.run_native_refresh_after_wiki_integration(
        root,
        state,
        workdir=ROOT,
        reason="threshold",
    )

    assert code == 18
    assert payload["failure"]["reason"] == "semantic-artifact-refresh-failed"
    assert payload["semantic_artifacts"] == report
    assert payload["runs"] == []
    assert payload["status_after"]["should_refresh"] is True
    assert len(batch_native_refresh.pending_entries(state)) == 1


def test_native_refresh_requeues_when_active_workspace_lacks_integrated_source(tmp_path: Path, monkeypatch) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "work" / "wikigraph" / "state"
    watched_dir = tmp_path / "watched"
    watched_dir.mkdir()
    batch_native_refresh.mark_pending(state, root, reason="wiki-integration:threshold")
    _set_native_cutover_env(monkeypatch, watched_dir)
    _stub_semantic_artifact_refresh(monkeypatch)

    def fake_refresh_cutover(**kwargs):
        batch_native_refresh.clear_pending(kwargs["state_dir"])
        return {"cutover": True, "refresh_kind": "incremental", "status_after": batch_native_refresh.status(root, state)}

    monkeypatch.setattr(batch_wiki_integration.batch_native_refresh, "refresh_cutover", fake_refresh_cutover)
    monkeypatch.setattr(
        batch_wiki_integration.native_semantic_artifact_refresh,
        "validate_active_workspace_coverage",
        lambda *args, **kwargs: {
            "ok": False,
            "workspace_id": "active-stale",
            "covered_path_count": 0,
            "failures": [{"code": "active-workspace-source-coverage-missing"}],
        },
    )

    code, payload = batch_wiki_integration.run_native_refresh_after_wiki_integration(
        root,
        state,
        workdir=ROOT,
        reason="threshold",
        max_passes=1,
    )

    assert code == 19
    assert payload["failure"]["reason"] == "active-workspace-semantic-coverage-failed"
    assert payload["active_workspace_coverage"]["ok"] is False
    assert payload["status_after"]["should_refresh"] is True
    assert len(batch_native_refresh.pending_entries(state)) == 1
    assert batch_native_refresh.pending_entries(state)[0]["reason"].startswith("wiki-integration:")


@pytest.mark.subprocess
def test_batch_wiki_integration_auto_integrate_records_failure_if_runner_leaves_ledger_pending(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "work" / "wikigraph" / "state"
    mark_pending_batch(state, root, 10, path_prefix="260107")
    noop_runner = tmp_path / "noop_integrator.py"
    write(noop_runner, "print('noop integration runner returned without clearing ledger')\n")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ops.batch_wiki_integration",
            "auto-integrate",
            "--root",
            str(root),
            "--state-dir",
            str(state),
            "--reason",
            "threshold",
            "--integration-command",
            f"{sys.executable} {noop_runner}",
        ],
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)

    assert result.returncode == 12
    assert payload["ran"] is True
    assert payload["post_status"]["should_integrate"] is True
    assert payload["failure"]["reason"] == "auto-integrate-incomplete"
    assert load_pending_wiki_integration_ledger(state)["last_failed_integration"]["reason"] == "auto-integrate-incomplete"


@pytest.mark.subprocess
def test_batch_wiki_integration_cli_status_mark_and_clear_are_external(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "work" / "wikigraph" / "state"
    module_cmd = [sys.executable, "-m", "ops.batch_wiki_integration"]

    mark = subprocess.run(
        [
            *module_cmd,
            "mark-pending",
            "--root",
            str(root),
            "--state-dir",
            str(state),
            "--raw-path",
            "raw/clip/2601/26010101_Foo-Paper.md",
            "--title",
            "Foo Paper",
            "--source-id",
            "https://arxiv.org/abs/2601.0101",
            "--topic-hint",
            "agents",
            "--required-section",
            "methodology",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    mark_payload = json.loads(mark.stdout)
    assert mark_payload["pending_count"] == 1
    assert mark_payload["threshold"] == DEFAULT_PENDING_WIKI_INTEGRATION_THRESHOLD
    assert mark_payload["pending"][0]["topic_hints"] == ["agents"]
    assert mark_payload["pending"][0]["required_sections"] == ["methodology"]

    status = subprocess.run([*module_cmd, "status", "--root", str(root), "--state-dir", str(state)], check=True, text=True, capture_output=True)
    assert json.loads(status.stdout)["should_integrate"] is False
    assert not (root / "pending_wiki_integration.json").exists()

    clear = subprocess.run(
        [*module_cmd, "clear-success", "--root", str(root), "--state-dir", str(state), "--integrated-path", "raw/clip/2601/26010101_Foo-Paper.md"],
        check=True,
        text=True,
        capture_output=True,
    )
    assert json.loads(clear.stdout)["cleared_count"] == 1
    assert load_pending_wiki_integration_ledger(state)["pending"] == []


def test_refresh_native_after_integration_cli_uses_machine_repair_chain(tmp_path: Path, monkeypatch, capsys) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "state"
    calls = []

    def fake_followthrough(root_arg, state_arg, *, workdir, reason, max_passes, allow_embedding_contract_change, embedding_profile=None):
        calls.append((root_arg, state_arg, workdir, reason, max_passes, allow_embedding_contract_change))
        return 0, {"native_refresh": True, "semantic_artifacts": {"ok": True}, "status_after": {"should_refresh": False}}

    monkeypatch.setattr(batch_wiki_integration, "run_native_refresh_after_wiki_integration", fake_followthrough)

    code = batch_wiki_integration.main(
        [
            "refresh-native-after-integration",
            "--root",
            str(root),
            "--state-dir",
            str(state),
            "--workdir",
            str(ROOT),
            "--reason",
            "semantic-retry",
            "--max-passes",
            "3",
            "--allow-embedding-contract-change",
        ]
    )

    assert code == 0
    assert json.loads(capsys.readouterr().out)["semantic_artifacts"]["ok"] is True
    assert calls == [(root.resolve(), state.resolve(), ROOT, "semantic-retry", 3, True)]


def _fail_if_native_refresh_runs(*args, **kwargs):
    raise AssertionError("native refresh pipeline must not run when defer_native_refresh is set")


def _prepare_native_refresh_fixture(tmp_path: Path, monkeypatch):
    root = sample_wiki(tmp_path)
    state = tmp_path / "work" / "wikigraph" / "state"
    watched_dir = tmp_path / "watched"
    watched_dir.mkdir()
    batch_native_refresh.mark_pending(state, root, reason="wiki-integration:threshold")
    _set_native_cutover_env(monkeypatch, watched_dir)
    _stub_semantic_artifact_refresh(monkeypatch)
    return root, state


def _stub_cutover_recording_profiles(monkeypatch, seen_profiles: list) -> None:
    def fake_refresh_cutover(**kwargs):
        seen_profiles.append(kwargs["embedding_profile"])
        batch_native_refresh.clear_pending(kwargs["state_dir"])
        return {
            "cutover": True,
            "skipped": False,
            "refresh_kind": "incremental",
            "fill_missing_vectors": True,
            "vector_cache_required": True,
            "policy_native_pending": False,
            "status_after": batch_native_refresh.status(kwargs["root"], kwargs["state_dir"]),
        }

    monkeypatch.setattr(batch_wiki_integration.batch_native_refresh, "refresh_cutover", fake_refresh_cutover)


@pytest.mark.parametrize(
    ("env_value", "flag", "expected"),
    [
        (None, "operator-fast", "operator-fast"),
        (None, None, "conservative"),
        ("balanced-medium", None, "balanced-medium"),
        ("balanced-medium", "operator-fast", "operator-fast"),
    ],
)
def test_native_refresh_threads_embedding_profile_flag_to_cutover(
    tmp_path: Path, monkeypatch, env_value, flag, expected
) -> None:
    if env_value is None:
        monkeypatch.delenv("LLM_WIKI_NATIVE_EMBEDDING_PROFILE", raising=False)
    else:
        monkeypatch.setenv("LLM_WIKI_NATIVE_EMBEDDING_PROFILE", env_value)
    root, state = _prepare_native_refresh_fixture(tmp_path, monkeypatch)
    seen = []
    _stub_cutover_recording_profiles(monkeypatch, seen)
    kwargs = {} if flag is None else {"embedding_profile": flag}

    code, payload = batch_wiki_integration.run_native_refresh_after_wiki_integration(
        root, state, workdir=ROOT, reason="threshold", **kwargs
    )

    assert code == 0
    assert seen == [expected]
    assert payload["embedding_profile"] == expected


def test_native_refresh_invalid_embedding_profile_fails_closed(tmp_path: Path, monkeypatch) -> None:
    root, state = _prepare_native_refresh_fixture(tmp_path, monkeypatch)
    _stub_cutover_recording_profiles(monkeypatch, [])
    cases = [
        {"embedding_profile": "turbo-nonsense"},
        "env",
    ]
    for case in cases:
        if case == "env":
            monkeypatch.setenv("LLM_WIKI_NATIVE_EMBEDDING_PROFILE", "turbo-nonsense")
            kwargs = {}
        else:
            kwargs = case
        with pytest.raises(ValueError, match="unknown embedding profile"):
            batch_wiki_integration.run_native_refresh_after_wiki_integration(
                root, state, workdir=ROOT, reason="threshold", **kwargs
            )


def test_refresh_native_after_integration_cli_accepts_embedding_profile(tmp_path: Path, monkeypatch, capsys) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "work" / "wikigraph" / "state"
    calls = []

    def fake_refresh(root_arg, state_arg, *, workdir, reason, max_passes, allow_embedding_contract_change, embedding_profile=None):
        calls.append(embedding_profile)
        return 0, {"ok": True}

    monkeypatch.setattr(batch_wiki_integration, "run_native_refresh_after_wiki_integration", fake_refresh)
    code = batch_wiki_integration.main(
        [
            "refresh-native-after-integration",
            "--root",
            str(root),
            "--state-dir",
            str(state),
            "--workdir",
            str(ROOT),
            "--embedding-profile",
            "balanced-medium",
        ]
    )

    assert code == 0
    assert calls == ["balanced-medium"]


def _forbid_native_refresh_pipeline(monkeypatch) -> None:
    monkeypatch.setattr(batch_wiki_integration.native_semantic_artifact_refresh, "refresh_semantic_artifacts", _fail_if_native_refresh_runs)
    monkeypatch.setattr(batch_wiki_integration.batch_native_refresh, "refresh_cutover", _fail_if_native_refresh_runs)


def _sample_index_with_meta(root: Path) -> None:
    write(
        root / "index.md",
        "# LLM Wiki Index\n\n> Last updated: 2026-05-18 16:00 | Total pages: 4\n\n"
        "## Concepts\n\n- [[foo]] - Foo page.\n\n"
        "## Queries\n\n- [[bar]] - Bar page.\n\n"
        "## Meta\n\n- [[raw-clip-map]] - Raw clip map.\n- [[topic-map]] - Topic map.\n",
    )


def test_auto_integrate_defer_native_refresh_skips_refresh_and_keeps_native_pending(tmp_path: Path, monkeypatch) -> None:
    root = sample_wiki(tmp_path)
    _sample_index_with_meta(root)
    state = tmp_path / "work" / "wikigraph" / "state"
    mark_pending_batch(
        state, root, 10, path_prefix="260109", topic_hints=["defer native refresh topic"], required_sections=["summary"]
    )
    _forbid_native_refresh_pipeline(monkeypatch)

    code, payload = batch_wiki_integration.run_auto_integration(root, state, reason="threshold", defer_native_refresh=True)

    assert code == 0
    native_refresh = payload["local_result"]["native_refresh"]
    assert native_refresh["native_refresh"] is True
    assert native_refresh["skip_reason"] == "defer_native_refresh"
    _assert_native_deferred(payload, state)
    assert load_pending_wiki_integration_ledger(state)["pending"] == []


def test_auto_integrate_external_runner_defer_native_refresh_skips_followthrough(tmp_path: Path, monkeypatch) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "work" / "wikigraph" / "state"
    mark_pending_batch(
        state, root, 10, path_prefix="260110", topic_hints=["defer native refresh topic"], required_sections=["summary"]
    )
    fake_runner = tmp_path / "fake_wiki_integrator.py"
    _write_fake_integrator(fake_runner)
    _forbid_native_refresh_pipeline(monkeypatch)

    code, payload = batch_wiki_integration.run_auto_integration(
        root,
        state,
        reason="threshold",
        integration_command=f"{sys.executable} {fake_runner}",
        defer_native_refresh=True,
    )

    assert code == 0
    _assert_native_deferred(payload, state)


def test_mark_pending_cli_auto_integrate_defer_native_refresh(tmp_path: Path, monkeypatch, capsys) -> None:
    root = sample_wiki(tmp_path)
    _sample_index_with_meta(root)
    state = tmp_path / "work" / "wikigraph" / "state"
    mark_pending_batch(
        state, root, 9, path_prefix="260111", topic_hints=["defer native refresh topic"], required_sections=["summary"]
    )
    _forbid_native_refresh_pipeline(monkeypatch)

    code = batch_wiki_integration.main(
        [
            "mark-pending",
            "--root",
            str(root),
            "--state-dir",
            str(state),
            "--raw-path",
            "raw/clip/2601/26011109_Paper.md",
            "--title",
            "Paper 9",
            "--topic-hint",
            "defer native refresh topic",
            "--required-section",
            "summary",
            "--auto-integrate",
            "--defer-native-refresh",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    _assert_native_deferred(payload, state, nested=True)


def test_integrate_local_cli_defer_native_refresh(tmp_path: Path, monkeypatch, capsys) -> None:
    root = sample_wiki(tmp_path)
    _sample_index_with_meta(root)
    state = tmp_path / "work" / "wikigraph" / "state"
    mark_pending_batch(
        state, root, 10, path_prefix="260112", topic_hints=["defer native refresh topic"], required_sections=["summary"]
    )
    _forbid_native_refresh_pipeline(monkeypatch)

    code = batch_wiki_integration.main(
        [
            "integrate-local",
            "--root",
            str(root),
            "--state-dir",
            str(state),
            "--defer-native-refresh",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    _assert_native_deferred(payload, state)
