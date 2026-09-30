"""`remote/trial.py` を参加者が打つとおりに走らせる。

偽の worker を立て、`--endpoint` でそこへ向ける。simulator は要らない。
見たいのは、参加者の入り口が最初から最後まで繋がっているか——Agent の読み込み、
接続、`step`、結果の表示。

サブプロセスで走らせるのは、参加者が実際にやるのと同じ形にするため。
同一プロセスだと `sys.modules` が汚れていて、import の抜けを見逃す。
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from capx.agent_api import StepResult, TaskSpec
from capx.local_env import TrialOutcome
from capx.remote_env.worker.server import WorkerServer

REPO = Path(__file__).resolve().parent.parent
TRIAL = REPO / "remote" / "trial.py"
EXAMPLE = REPO / "remote" / "agents" / "example_agent.py"


class FakeEnv:
    def __init__(self) -> None:
        self.codes: list[str] = []
        self.seeds: list[int | None] = []

    def step(self, code: str, *, capture_video: bool = False) -> StepResult:
        self.codes.append(code)
        return StepResult(request_id="r", ok=True, stdout="", stderr="")

    def render(self, camera: str = "main") -> bytes:
        return b"\xff\xd8JPEG"

    def reset_for_trial(self, trial: int, seed: int | None = None) -> TaskSpec:
        self.seeds.append(seed)
        return TaskSpec(
            task_id="fake",
            seed=trial,
            instruction="\nGoal: stack the cube.",
            api_docs="",
        )

    def evaluate(self) -> TrialOutcome:
        return TrialOutcome(
            reward=0.25,
            task_completed=False,
            terminated=False,
            truncated=False,
            sandbox_rc=0,
        )


@pytest.fixture
def worker():
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    env = FakeEnv()
    server = WorkerServer(env, session_id="s")
    threading.Thread(
        target=server.serve,
        kwargs={"port": port, "host": "127.0.0.1"},
        daemon=True,
    ).start()
    time.sleep(0.4)
    yield env, f"tcp://127.0.0.1:{port}"
    server._stop.set()


def _run(*args: str, env_extra: dict | None = None) -> subprocess.CompletedProcess:
    import os

    env = {**os.environ, **(env_extra or {})}
    env.pop("CAPX_ENV_SERVER_URL", None) if not env_extra else None
    return subprocess.run(
        [sys.executable, str(TRIAL), *args],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
        cwd=REPO,
    )


def test_example_agent_runs_end_to_end(worker) -> None:
    env, endpoint = worker
    out = _run("--agent", str(EXAMPLE), "--endpoint", endpoint)

    assert out.returncode == 0, out.stderr
    assert env.codes == ["open_gripper()"], "Agent のコードが worker に届いていない"
    # 参加者が見る結果
    assert "task_completed" in out.stdout
    assert "reward" in out.stdout
    assert "0.2500" in out.stdout
    assert "steps_used     : 1" in out.stdout
    # 参加者が最初に読む example_agent が、渡されたタスクを実際に表示できる
    assert "Goal: stack the cube." in out.stdout


def test_result_is_shown_to_the_human_not_the_agent(worker) -> None:
    """Agent の出力に reward が出ていないこと。

    example_agent は `env.step()` の結果と画像サイズしか表示しない。
    reward は trial.py が最後に出すもので、Agent の print には混ざらない。
    """
    _env, endpoint = worker
    out = _run("--agent", str(EXAMPLE), "--endpoint", endpoint)
    before_result = out.stdout.split("trial 1")[0]
    assert "reward" not in before_result


def test_agent_path_relative_to_remote_dir_works(worker) -> None:
    """`remote/` の中でも外でも、同じ書き方で見つかる。"""
    _env, endpoint = worker
    out = _run("--agent", "agents/example_agent.py", "--endpoint", endpoint)
    assert out.returncode == 0, out.stderr


def test_seed_reaches_the_worker(worker) -> None:
    env, endpoint = worker
    _run("--agent", str(EXAMPLE), "--endpoint", endpoint, "--seed", "7")
    assert env.seeds == [7]


def test_multiple_trials_show_a_mean(worker) -> None:
    env, endpoint = worker
    out = _run(
        "--agent", str(EXAMPLE), "--endpoint", endpoint, "--total-trials", "3"
    )
    assert out.returncode == 0, out.stderr
    assert len(env.codes) == 3
    assert "3 trial" in out.stdout


def test_missing_agent_says_where_it_looked() -> None:
    out = _run("--agent", "nope/missing.py", "--endpoint", "tcp://127.0.0.1:1")
    assert out.returncode != 0
    assert "見つからない" in out.stderr + out.stdout
    assert "nope/missing.py" in out.stderr + out.stdout


def test_no_server_configured_says_what_to_set() -> None:
    out = subprocess.run(
        [sys.executable, str(TRIAL), "--agent", str(EXAMPLE)],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=REPO,
        env={k: v for k, v in __import__("os").environ.items() if k != "CAPX_ENV_SERVER_URL"},
    )
    assert out.returncode != 0
    assert "CAPX_ENV_SERVER_URL" in out.stderr + out.stdout
