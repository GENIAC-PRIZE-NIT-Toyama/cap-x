"""`remote/tracing.py`。Langfuse には送らず、手元に溜めた span の親子と属性を見る。"""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from capx.agent_api import BudgetExceeded, EnvUnavailable, StepResult, TaskSpec

REMOTE = Path(__file__).resolve().parent.parent / "remote"
if str(REMOTE) not in sys.path:
    sys.path.insert(0, str(REMOTE))

import tracing  # noqa: E402


class FakeEnv:
    def __init__(self) -> None:
        self.closed = False
        self.recorded_steps = ["s"]

    def step(self, code: str, *, capture_video: bool = False) -> StepResult:
        if code == "boom":
            raise EnvUnavailable("切れた")
        return StepResult(request_id="r", ok=code != "bad", stdout="out", stderr="" if code != "bad" else "err")

    def render(self, camera: str = "main") -> bytes:
        return b"jpeg"

    def reset_for_trial(self, trial: int, seed: int | None = None) -> TaskSpec:
        return TaskSpec(task_id="cube_stack", seed=seed, instruction="Goal: stack.", api_docs="docs")


def _summary(**kw):
    result = SimpleNamespace(failure=kw.pop("failure", None))
    base = dict(reward=0.5, task_completed=False, success=True, num_code_blocks=2, code_path=None, result=result)
    return SimpleNamespace(**{**base, **kw})


def _start(client, **kw):
    args = dict(
        session="m/cube_stack/20261022_000000",
        task="cube_stack",
        trial=1,
        seed=7,
        attempt=1,
        model="m",
        agent_path=__file__,
        tags=["trial"],
        infra_errors=(EnvUnavailable,),
    )
    return tracing.trial_trace(**{**args, **kw})


@pytest.fixture
def client(monkeypatch):
    """手元に span を溜める Langfuse。score は送らずに控える。"""
    pytest.importorskip("langfuse")
    from langfuse import Langfuse
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    # 送り先の準備は public key ごとに 1 回しか作られないので、テストごとに変える
    lf = Langfuse(
        public_key=f"pk-test-{uuid.uuid4().hex}",
        secret_key="sk-test",
        base_url="http://127.0.0.1:9",
        tracer_provider=TracerProvider(),
        span_exporter=exporter,
    )
    scores: list[dict] = []
    monkeypatch.setattr(lf, "create_score", lambda **kw: scores.append(kw))
    monkeypatch.setattr(tracing, "_client", lf)
    monkeypatch.setenv("CAPX_USER", "tester")

    def spans():
        lf.flush()
        return {s.name: s for s in exporter.get_finished_spans()}

    return SimpleNamespace(lf=lf, spans=spans, scores=scores)


def test_without_keys_nothing_is_wrapped(monkeypatch) -> None:
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.setattr(tracing, "_client", None)
    assert tracing.setup() is False
    env = FakeEnv()
    with _start(None) as trace:
        assert trace.wrap(env) is env
        trace.finish(_summary())  # 何もしない


def test_no_trace_flag_wins_over_keys(monkeypatch) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    monkeypatch.setattr(tracing, "_client", None)
    assert tracing.setup(enabled=False) is False


def test_steps_are_children_of_the_trial(client) -> None:
    with _start(client) as trace:
        env = trace.wrap(FakeEnv())
        env.reset_for_trial(1, 7)
        env.step("print(1)")
        env.step("bad")
        assert env.recorded_steps == ["s"], "Bench が使う属性はそのまま通す"
        trace.finish(_summary())

    spans = client.spans()
    root, step = spans["cube_stack"], spans["env.step"]
    assert step.parent.span_id == root.context.span_id
    assert step.attributes["langfuse.observation.type"] == "tool"
    assert root.attributes["user.id"] == "tester"
    assert root.attributes["session.id"] == "m/cube_stack/20261022_000000"
    task_input = json.loads(root.attributes["langfuse.observation.input"])
    assert task_input["instruction"] == "Goal: stack." and task_input["seed"] == 7
    assert {s["name"] for s in client.scores} == {"reward", "task_completed", "exec_ok"}


def test_failure_kind_is_scored(client) -> None:
    with _start(client) as trace:
        trace.finish(_summary(failure={"kind": "budget", "message": "上限"}))
    assert {"name": "failure_kind", "value": "budget"}.items() <= next(
        s for s in client.scores if s["name"] == "failure_kind"
    ).items()


def test_budget_exceeded_is_recorded_and_reraised(client) -> None:
    class Limited(FakeEnv):
        def step(self, code, *, capture_video=False):
            raise BudgetExceeded("execution_budget", "上限")

    with _start(client) as trace:
        with pytest.raises(BudgetExceeded):
            trace.wrap(Limited()).step("x")
    assert client.spans()["env.step"].attributes["langfuse.observation.level"] == "WARNING"


def test_infra_failure_marks_the_attempt(client) -> None:
    with pytest.raises(EnvUnavailable):
        with _start(client) as trace:
            trace.wrap(FakeEnv()).step("boom")
    root = client.spans()["cube_stack"]
    assert root.attributes["langfuse.observation.level"] == "ERROR"
    assert tracing.INFRA_FAILED_TAG in root.attributes["langfuse.trace.tags"]
    assert client.scores == [], "やり直した attempt には score を付けない"


def test_non_ascii_user_is_dropped(monkeypatch, capsys) -> None:
    monkeypatch.setenv("CAPX_USER", "山本")
    monkeypatch.setattr(tracing, "_warned_user", False)
    assert tracing._user_id() is None
    assert "ローマ字" in capsys.readouterr().out


def test_session_follows_the_run_dir(tmp_path) -> None:
    run_dir = tmp_path / "qwen_qwen3.5-9b" / "suite_quick" / "20261022_141636"
    assert tracing.session_id(run_dir, "qwen/qwen3.5-9b", "x") == "qwen_qwen3.5-9b/suite_quick/20261022_141636"
    assert tracing.session_id(None, "qwen/qwen3.5-9b", "cube_stack").startswith("qwen_qwen3.5-9b/cube_stack/")


def test_video_falls_back_to_turn_videos(tmp_path) -> None:
    pytest.importorskip("langfuse")
    trial_dir = tmp_path / "trial_01"
    (trial_dir / "videos").mkdir(parents=True)
    summary = _summary(code_path=str(trial_dir / "code.py"))
    assert tracing._video(summary) is None

    (trial_dir / "videos" / "turn_01.mp4").write_bytes(b"one")
    assert not isinstance(tracing._video(summary), list), "1 本ならそのまま"
    (trial_dir / "videos" / "turn_02.mp4").write_bytes(b"two")
    assert len(tracing._video(summary)) == 2, "つないだ動画が無ければターンごと"
    (trial_dir / "videos" / "combined.mp4").write_bytes(b"all")
    assert tracing._video(summary)._content_bytes == b"all"


def test_code_is_shown_as_a_code_block() -> None:
    # Langfuse の UI は文字列を Markdown で表示する。`#` が見出しにならないよう囲む
    assert tracing._code_block("# 観測\nx = 1", "python") == "```python\n# 観測\nx = 1\n```"
    assert tracing._code_block("") == ""
    assert tracing._code_block("s = '```'\n").startswith("````\n"), "中の ``` で閉じない"


@pytest.mark.parametrize("step_videos", [False, True])
def test_step_video_only_when_asked(client, monkeypatch, step_videos) -> None:
    class WithVideo(FakeEnv):
        def step(self, code, *, capture_video=False):
            return StepResult(request_id="r", ok=True, stdout="", stderr="", video=b"mp4", video_media_type="video/mp4")

    monkeypatch.setattr(tracing, "_step_videos", step_videos)
    with _start(client) as trace:
        trace.wrap(WithVideo()).step("x")
    output = client.spans()["env.step"].attributes["langfuse.observation.output"]
    assert ("@@@langfuseMedia:type=video/mp4" in output) is step_videos


def test_llm_outage_is_not_scored(client) -> None:
    # trial.py は LLM の障害でやり直さないが、Agent の失敗としては数えない（suite.py と同じ）
    with _start(client) as trace:
        trace.finish(_summary(failure={"kind": "llm_timeout", "message": "応答がない"}))
    root = client.spans()["cube_stack"]
    assert client.scores == []
    assert tracing.INFRA_FAILED_TAG in root.attributes["langfuse.trace.tags"]
    assert root.attributes["langfuse.observation.level"] == "ERROR"


def test_tracing_trouble_does_not_stop_the_trial(client, monkeypatch, capsys) -> None:
    def broken(summary):
        raise OSError("動画を読めない")

    monkeypatch.setattr(tracing, "_video", broken)
    with _start(client) as trace:
        trace.finish(_summary())
    assert "試行は続ける" in capsys.readouterr().out


def test_non_ascii_tags_are_kept(client) -> None:
    with _start(client, tags=["suite:持ち上げ"]):
        pass
    assert "suite:____" in client.spans()["cube_stack"].attributes["langfuse.trace.tags"]
