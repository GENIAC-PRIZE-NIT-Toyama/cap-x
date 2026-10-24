"""`remote/trial.py` を参加者が打つとおりに走らせる。

偽の worker を立て、`--endpoint` でそこへ向ける。simulator は要らない。
見たいのは、参加者の入り口が最初から最後まで繋がっているか——Agent の読み込み、
接続、`step`、結果の表示。

サブプロセスで走らせるのは、参加者が実際にやるのと同じ形にするため。
同一プロセスだと `sys.modules` が汚れていて、import の抜けを見逃す。
"""

from __future__ import annotations

import importlib.util
import os
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
# 手元の remote/.env を読ませない（python-dotenv が無い環境では、もともと読まれない）
NO_DOTENV = ["--env-file", os.devnull] if importlib.util.find_spec("dotenv") else []


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
    """保存先を指定しない呼び出しは `--no-save` にする（テストで `remote/outputs/` を汚さない）。"""
    env = {**os.environ, **(env_extra or {})}
    env.pop("CAPX_ENV_SERVER_URL", None) if not env_extra else None
    if "--output-dir" not in args:
        args = (*args, "--no-save")
    return subprocess.run(
        [sys.executable, str(TRIAL), *NO_DOTENV, *args],
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
        [sys.executable, str(TRIAL), *NO_DOTENV, "--agent", str(EXAMPLE)],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=REPO,
        env={k: v for k, v in os.environ.items() if k != "CAPX_ENV_SERVER_URL"},
    )
    assert out.returncode != 0
    assert "CAPX_ENV_SERVER_URL" in out.stderr + out.stdout


def test_results_go_under_model_task_and_run_time(worker, tmp_path) -> None:
    """`<保存先>/<モデル名>/<タスク名>/<日時>/trial_01_.../`。

    偽の worker は close で止まる（1 セッション 1 worker）ので、ここでは 1 回だけ実行する。
    実行ごとに分かれることは下の `_make_run_dir` のテストで見る。
    """
    _env, endpoint = worker
    out = _run(
        "--agent", str(EXAMPLE), "--endpoint", endpoint,
        "--model", "vendor/some-model", "--output-dir", str(tmp_path),
    )
    assert out.returncode == 0, out.stderr

    runs = list((tmp_path / "vendor_some-model" / "cube_stack").iterdir())
    assert len(runs) == 1
    trials = list(runs[0].glob("trial_01_*"))
    assert len(trials) == 1 and (trials[0] / "result.json").exists()
    assert str(runs[0]) in out.stdout, "保存先を表示する"


def _trial_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("remote_trial", TRIAL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_root_is_remote_outputs_wherever_it_is_run_from() -> None:
    """ローカル実行の `outputs/`（リポジトリ直下）と混ざらない。"""
    assert _trial_module().DEFAULT_OUTPUT_ROOT == REPO / "remote" / "outputs"


def test_each_run_gets_its_own_folder(tmp_path) -> None:
    """実行ごとに日時のフォルダが分かれ、前の結果と混ざらない。同じ秒でも分かれる。"""
    make = _trial_module()._make_run_dir
    first = make(str(tmp_path), "m", "t", now=0)
    later = make(str(tmp_path), "m", "t", now=60)
    same_second = make(str(tmp_path), "m", "t", now=0)
    assert len({first, later, same_second}) == 3
    assert same_second.name == f"{first.name}_2"
