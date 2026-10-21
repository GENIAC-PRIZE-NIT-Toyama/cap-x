"""`remote/suite.py`（評価スイート）と `remote/suite_summary.py`（集計）。

GPU マシンは使わない。失敗の振り分けは `run_trial` を差し替えて見て、入り口は
`test_remote_trial.py` と同じく偽の worker に `--endpoint` で向けて走らせる。
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from capx.agent_api import BudgetExceeded, EnvUnavailable, StepResult, TaskSpec
from capx.local_env import TrialOutcome
from capx.remote_env.worker.server import WorkerServer

REPO = Path(__file__).resolve().parent.parent
REMOTE = REPO / "remote"
sys.path.insert(0, str(REMOTE))

import suite  # noqa: E402
import suite_summary  # noqa: E402


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


# -- 設定ファイル ----------------------------------------------------------


def test_minimal_suite_gets_defaults(tmp_path) -> None:
    config = suite.load_suite(_write(tmp_path / "quick.yaml", "tasks: [cube_stack]\n"))

    assert config.name == "quick", "name を省略するとファイル名"
    assert config.trials == 1
    assert config.seed is None, "seed を省略するとランダム"
    assert config.max_steps == 10, "max_steps を省略すると worker の既定と同じ"


def test_all_problems_are_reported_at_once(tmp_path) -> None:
    path = _write(
        tmp_path / "bad.yaml",
        "trials: 0\nseed: true\nmax_steps: 21\nmax_step: 3\ntasks: [a, b, a]\n",
    )
    with pytest.raises(suite.SuiteConfigError) as err:
        suite.load_suite(path)

    message = str(err.value)
    assert "max_step" in message, "打ち間違えたキーを黙って無視しない"
    assert "trials" in message
    assert "seed" in message, "bool は整数として通さない"
    assert "max_steps は 1〜20" in message
    assert "a" in message and "2 回以上" in message


def test_saved_suite_reads_back_the_same(tmp_path) -> None:
    config = suite.load_suite(_write(tmp_path / "s.yaml", "tasks: [x, y]\ntrials: 3\nmax_steps: 4\n"))
    again = suite.load_suite(_write(tmp_path / "copy.yaml", config.to_yaml()))
    assert again == suite.SuiteConfig(name="s", tasks=("x", "y"), trials=3, seed=None, max_steps=4)


# -- 登録済みのスイート ------------------------------------------------------


def _count(tasks, prefix: str) -> int:
    return sum(t.startswith(prefix) for t in tasks)


@pytest.mark.parametrize("name", ["quick", "full"])
def test_bundled_suites_name_tasks_the_server_has(name) -> None:
    from capx.remote_env.server.tasks import TASKS

    config = suite.load_suite(REMOTE / "suites" / f"{name}.yaml")
    assert config.name == name
    assert set(config.tasks) <= set(TASKS)


def test_full_and_quick_suites_have_the_agreed_tasks() -> None:
    from capx.remote_env.server.tasks import TASKS

    full = suite.load_suite(REMOTE / "suites" / "full.yaml").tasks
    quick = suite.load_suite(REMOTE / "suites" / "quick.yaml").tasks
    robosuite = {t for t in TASKS if not t.startswith("libero_")}

    assert set(robosuite) <= set(full), "robosuite は全タスク"
    assert (_count(full, "libero_90_"), _count(full, "libero_10_")) == (20, 10)
    assert len(full) == len(robosuite) + 30, "LIBERO-Spatial / Object / Goal は入れない"

    assert set(quick) <= set(full), "quick は full の部分集合"
    assert (len(quick) - _count(quick, "libero_"), _count(quick, "libero_90_"), _count(quick, "libero_10_")) == (3, 5, 2)
    assert len(quick) == 10


# -- max_steps ------------------------------------------------------------


class _CountingEnv:
    def __init__(self) -> None:
        self.codes: list[str] = []

    def step(self, code, *, capture_video=False):
        self.codes.append(code)
        return StepResult(request_id="r", ok=True, stdout="", stderr="")

    def reset_for_trial(self, trial, seed=None):
        return "task"

    recorded_steps = ["passed through"]


def test_step_limit_stops_before_reaching_the_gpu_machine() -> None:
    inner = _CountingEnv()
    env = suite.StepLimitedEnv(inner, max_steps=2)

    env.step("a")
    env.step("b")
    with pytest.raises(BudgetExceeded):
        env.step("c")
    assert inner.codes == ["a", "b"], "上限を超えた step は送らない"

    env.reset_for_trial(2)
    env.step("d")
    assert inner.codes[-1] == "d", "試行が変わったら数え直す"
    assert env.recorded_steps == ["passed through"], "Bench が使うものはそのまま通す"


# -- 失敗の振り分け --------------------------------------------------------


class _Env:
    closed = 0

    def close(self) -> None:
        _Env.closed += 1


def _summary(tmp: Path, trial: int, kind: str | None = None):
    """`run_trial` の返り値の代わり。試行フォルダも作る（捨てるときに動かすため）。"""
    trial_dir = tmp / f"trial_{trial:02d}_sandboxrc_0_reward_0.000_taskcompleted_0"
    trial_dir.mkdir(parents=True, exist_ok=True)
    (trial_dir / "code.py").write_text("")
    (trial_dir / "result.json").write_text("{}")
    failure = {"kind": kind, "message": f"{kind} が起きた"} if kind else None
    return SimpleNamespace(
        task_completed=False,
        num_code_blocks=1,
        code_path=str(trial_dir / "code.py"),
        result=SimpleNamespace(failure=failure, wall_clock_s=1.0),
    )


@pytest.fixture
def runner(tmp_path, monkeypatch):
    """`run_trial` が返すもの（または投げるもの）を順に決められる Runner。"""
    plan: list = []
    calls: list[dict] = []

    def fake_run_trial(env, agent, trial, config, budget, **kw):
        calls.append({"env": env, "budget": budget, **kw})
        outcome = plan.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return _summary(Path(config["output_dir"]), trial, outcome)

    opens: list = []  # セッションを開くときに投げるもの。空なら開ける

    def fake_open_env(*a, **kw):
        if opens:
            raise opens.pop(0)
        return _Env()

    monkeypatch.setattr(suite, "run_trial", fake_run_trial)
    monkeypatch.setattr(suite, "load_agent", lambda *a, **kw: object())
    monkeypatch.setattr(suite, "_open_env", fake_open_env)

    agent = _write(tmp_path / "agent.py", "class Agent: ...\n")
    waits: list[float] = []
    r = suite.Runner(
        suite=suite.SuiteConfig(name="t", tasks=("cube_stack",), trials=1, seed=7, max_steps=3),
        run_dir=tmp_path,
        agent_path=agent,
        agent_sha256=suite._sha256(agent),
        model="m",
        record_video=False,
        server_url=None,
        endpoint="tcp://unused",
        sleep=waits.append,
    )
    (tmp_path / "cube_stack").mkdir()
    return SimpleNamespace(
        runner=r, plan=plan, opens=opens, calls=calls, waits=waits, agent=agent, task_dir=tmp_path / "cube_stack"
    )


def _events(task_dir: Path) -> list[dict]:
    return suite_summary.read_infra_events(task_dir)


def test_inference_failure_is_kept_as_a_result(runner) -> None:
    runner.plan.extend(["agent_error"])
    line = runner.runner.run_one({"env": None}, "cube_stack", 1)

    assert "[agent_error]" in line
    assert runner.waits == [], "推論の失敗はやり直さない"
    assert suite_summary.trial_dirs(runner.task_dir), "結果として残る"
    assert runner.calls[0]["budget"].max_steps == 3, "result.json に実際の予算が残る"
    assert runner.calls[0]["seed"] == 7


def test_gpu_failure_retries_with_a_new_session(runner) -> None:
    runner.plan.extend([EnvUnavailable("切れた"), None])
    holder = {"env": None}
    closed = _Env.closed

    line = runner.runner.run_one(holder, "cube_stack", 1)

    assert runner.waits == [10.0]
    assert _Env.closed == closed + 1, "中断した環境は使い回さない"
    assert runner.calls[1]["infrastructure_retries"] == 1
    assert runner.calls[0]["env"] is not runner.calls[1]["env"]
    assert [e["category"] for e in _events(runner.task_dir)] == ["gpu"]
    assert "infra retry 1: infrastructure" in line


def test_llm_timeout_retries_and_discards_the_attempt(runner) -> None:
    runner.plan.extend(["llm_timeout", None])
    line = runner.runner.run_one({"env": None}, "cube_stack", 1)

    assert runner.waits == [10.0]
    assert len(list((runner.task_dir / "_discarded").iterdir())) == 1, "調べられるように残す"
    assert "llm_timeout" in line
    assert [e["kind"] for e in _events(runner.task_dir)] == ["llm_timeout"]
    assert runner.calls[0]["seed"] == runner.calls[1]["seed"], "やり直しでも同じ seed"


def test_gives_up_after_the_retry_limit(runner) -> None:
    runner.plan.extend([EnvUnavailable("切れた")] * 4)
    line = runner.runner.run_one({"env": None}, "cube_stack", 1)

    assert line.startswith("infra_failed")
    assert runner.waits == [10.0, 20.0, 40.0]
    assert [e["gave_up"] for e in _events(runner.task_dir)] == [False, False, False, True]
    assert not suite_summary.trial_dirs(runner.task_dir), "結果は残らない（分母から外す）"


def test_config_failure_stops_the_suite(runner) -> None:
    runner.plan.extend(["llm_auth"])
    with pytest.raises(suite.SuiteAbortError):
        runner.runner.run_one({"env": None}, "cube_stack", 1)
    assert not suite_summary.trial_dirs(runner.task_dir), "止めた試行の結果は残さない"


def test_dropped_connection_while_opening_a_session_is_retried(runner) -> None:
    import requests

    runner.opens.append(requests.ConnectionError("reset"))
    runner.plan.extend([None])
    runner.runner.run_one({"env": None}, "cube_stack", 1)

    assert runner.waits == [10.0], "落ちずに GPU 側の失敗としてやり直す"
    assert [e["category"] for e in _events(runner.task_dir)] == ["gpu"]


@pytest.mark.parametrize("code", [401, 403, 404])
def test_session_refused_with_4xx_stops_the_suite(runner, code) -> None:
    from capx.agent_api import EnvStartFailed

    runner.opens.append(EnvStartFailed("断られた", status_code=code))
    with pytest.raises(suite.SuiteAbortError):
        runner.runner.run_one({"env": None}, "cube_stack", 1)
    assert runner.waits == [], "直らない失敗は待たない"


def test_session_failed_with_5xx_is_retried(runner) -> None:
    from capx.agent_api import EnvStartFailed

    runner.opens.append(EnvStartFailed("起動できない", status_code=503))
    runner.plan.extend([None])
    runner.runner.run_one({"env": None}, "cube_stack", 1)
    assert runner.waits == [10.0]


def test_llm_unreachable_is_retried_then_stops_the_suite(runner) -> None:
    runner.plan.extend(["llm_unreachable", None])
    runner.runner.run_one({"env": None}, "cube_stack", 1)
    assert runner.waits == [10.0], "一瞬の接続断ではスイートを止めない"

    runner.plan.extend(["llm_unreachable"] * 4)
    with pytest.raises(suite.SuiteAbortError):
        runner.runner.run_one({"env": None}, "cube_stack", 2)


def test_agent_edited_mid_run_stops_the_suite(runner) -> None:
    runner.agent.write_text("class Agent: changed\n")
    with pytest.raises(suite.SuiteAbortError, match="書き換わった"):
        runner.runner.run_one({"env": None}, "cube_stack", 1)
    assert runner.calls == [], "書き換わった Agent では回さない"


def test_agent_failing_to_load_stops_the_suite(runner, monkeypatch) -> None:
    def broken(*a, **kw):
        raise ImportError("no module named foo")

    monkeypatch.setattr(suite, "load_agent", broken)
    with pytest.raises(suite.SuiteAbortError, match="読み込めない"):
        runner.runner.run_one({"env": None}, "cube_stack", 1)


def test_discarding_twice_keeps_both(runner) -> None:
    runner.plan.extend(["llm_timeout", "llm_timeout", None])
    runner.runner.run_one({"env": None}, "cube_stack", 1)
    assert len(list((runner.task_dir / "_discarded").iterdir())) == 2, "前に捨てたものを上書きしない"


# -- 集計 ----------------------------------------------------------------


def _result(task_dir: Path, trial: int, *, completed, wall, steps_ok=(True,), failure=None, seed=1):
    d = task_dir / f"trial_{trial:02d}_sandboxrc_0_reward_0.000_taskcompleted_{int(bool(completed))}"
    d.mkdir(parents=True)
    _write(d / "result.json", json.dumps({
        "trial": trial,
        "seed": seed,
        "task_completed": completed,
        "wall_clock_s": wall,
        "execution_time_used_s": wall / 2,
        "failure": failure,
        "steps": [{"index": i, "ok": ok, "execution_time_s": 0.1} for i, ok in enumerate(steps_ok, 1)],
    }))


def _event(task_dir: Path, trial: int, *, gave_up: bool, category="gpu") -> None:
    suite._append_event(task_dir, {"trial": trial, "category": category, "kind": "x", "gave_up": gave_up})


@pytest.fixture
def run_dir(tmp_path) -> Path:
    config = suite.SuiteConfig(name="s", tasks=("cube_stack", "libero_object_3"), trials=3)
    _write(tmp_path / "suite.yaml", config.to_yaml())
    _write(tmp_path / "run.json", json.dumps({"agent": {"spec": "a.py", "model": "m"}}))

    cube = tmp_path / "cube_stack"
    # 成功 / 途中でエラーを出して直して成功 / 予算超過で未達成
    _result(cube, 1, completed=True, wall=10.0)
    _result(cube, 2, completed=True, wall=20.0, steps_ok=(False, True))
    _result(cube, 3, completed=False, wall=60.0, failure={"kind": "budget", "message": ""})

    libero = tmp_path / "libero_object_3"
    _result(libero, 1, completed=True, wall=5.0)
    # 試行 2: 1 回やり直して成功。試行 3: やり直しを使い切った
    _event(libero, 2, gave_up=False, category="llm")
    _result(libero, 2, completed=None, wall=7.0)
    for gave_up in (False, False, False, True):
        _event(libero, 3, gave_up=gave_up)
    # `_discarded/` の下は数えない
    _result(libero / "_discarded", 2, completed=True, wall=999.0)
    return tmp_path


def test_metrics_per_task(run_dir) -> None:
    s = suite_summary.summarize(run_dir)
    cube = s["tasks"]["cube_stack"]

    assert cube["pass_at_1"] == pytest.approx(2 / 3, abs=1e-4)
    assert cube["error_rate"] == pytest.approx(2 / 3, abs=1e-4), "成功しても途中のエラーは数える"
    assert cube["inference_errors"] == {"step_error": 1, "agent_error": 0, "budget": 1}
    assert cube["wall_clock_s"] == {"mean": 30.0, "median": 20.0, "p90": 52.0}


def test_infra_failures_are_left_out_of_the_denominator(run_dir) -> None:
    libero = suite_summary.summarize(run_dir)["tasks"]["libero_object_3"]

    assert libero["valid_trials"] == 2
    assert libero["infra"]["failed"] == [3]
    assert libero["infra"]["failure_rate"] == pytest.approx(1 / 3, abs=1e-4)
    assert libero["infra"]["retries"] == {"gpu": 3, "llm": 1}
    assert libero["unscored"] == 1
    assert libero["pass_at_1"] == 0.5, "採点できなかった試行は未達成"
    assert libero["wall_clock_s"]["median"] == 6.0, "_discarded の 999 秒は入らない"


def test_macro_micro_and_completeness(run_dir) -> None:
    s = suite_summary.summarize(run_dir)

    assert s["macro"]["pass_at_1"] == pytest.approx((2 / 3 + 0.5) / 2, abs=1e-4)
    assert s["micro"]["pass_at_1"] == pytest.approx(3 / 5, abs=1e-4)
    assert s["complete"] is True, "やり直しを使い切った試行も「終わった」"
    assert any("cube_stack" in n for n in s["notes"]), "robosuite の seed の注意"
    table = suite_summary.render(s)
    assert "infra_failed 1" in table
    assert "やり直し GPU 3 回, LLM 1 回" in table


def test_unfinished_trials_mark_the_run_incomplete(run_dir) -> None:
    import shutil

    shutil.rmtree(next((run_dir / "cube_stack").glob("trial_03_*")))
    s = suite_summary.summarize(run_dir)
    assert s["complete"] is False
    assert s["tasks"]["cube_stack"]["not_run"] == [3]
    assert "1 試行を回していない" in suite_summary.render(s)


# -- 入り口 ----------------------------------------------------------------


class _FakeEnv:
    def step(self, code, *, capture_video=False):
        return StepResult(request_id="r", ok=True, stdout="", stderr="")

    def render(self, camera="main"):
        return b"\xff\xd8JPEG"

    def reset_for_trial(self, trial, seed=None):
        return TaskSpec(task_id="fake", seed=seed or trial, instruction="Goal: x", api_docs="")

    def evaluate(self):
        return TrialOutcome(reward=1.0, task_completed=True, terminated=True, truncated=False, sandbox_rc=0)


@pytest.fixture
def endpoint():
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = WorkerServer(_FakeEnv(), session_id="s")
    threading.Thread(target=server.serve, kwargs={"port": port, "host": "127.0.0.1"}, daemon=True).start()
    time.sleep(0.4)
    yield f"tcp://127.0.0.1:{port}"
    server._stop.set()


def _run(*args: str) -> subprocess.CompletedProcess:
    import os

    env = {k: v for k, v in os.environ.items() if k != "CAPX_ENV_SERVER_URL"}
    return subprocess.run(
        [sys.executable, str(REMOTE / "suite.py"), *args],
        capture_output=True, text=True, timeout=120, env=env, cwd=REPO,
    )


def test_suite_runs_end_to_end(tmp_path, endpoint) -> None:
    # 2 回 step する Agent を max_steps 1 で回す → 2 回目で予算超過
    agent = _write(tmp_path / "two_steps.py", (
        "from capx.agent_api import AgentResult, BaseAgent\n"
        "class Agent(BaseAgent):\n"
        "    def run(self, env, task, budget):\n"
        "        env.step('a')\n"
        "        env.step('b')\n"
        "        return AgentResult()\n"
    ))
    suite_file = _write(tmp_path / "e2e.yaml", "tasks: [fake]\ntrials: 2\nseed: 5\nmax_steps: 1\n")
    out_dir = tmp_path / "out"

    out = _run(str(suite_file), "--agent", str(agent), "--endpoint", endpoint,
               "--output-dir", str(out_dir), "--model", "m", "--no-record-video")
    assert out.returncode == 0, out.stdout + out.stderr

    run_dir = next((out_dir / "m" / "suite_e2e").iterdir())
    summary = json.loads((run_dir / "summary.json").read_text())
    task = summary["tasks"]["fake"]
    assert summary["complete"] is True
    assert task["pass_at_1"] == 1.0
    assert task["inference_errors"]["budget"] == 2, "手元の max_steps が効いている"
    assert task["seeds"] == {"1": 5, "2": 5}
    assert (run_dir / "agent.py").read_text() == agent.read_text(), "使った Agent を残す"
    assert "pass@1" in out.stdout


def test_endpoint_takes_a_single_task_suite(tmp_path) -> None:
    suite_file = _write(tmp_path / "two.yaml", "tasks: [a, b]\n")
    out = _run(str(suite_file), "--endpoint", "tcp://127.0.0.1:1", "--output-dir", str(tmp_path))
    assert out.returncode != 0
    assert "1 つのスイートだけ" in out.stderr
