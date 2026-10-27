"""複数のタスクをまとめて評価する（評価スイート）。

    cd remote
    uv run suite.py suites/quick.yaml --agent agents/my_agent.py

スイートは YAML に書く（`suites/quick.yaml` / `suites/full.yaml`）。タスクを順に（`--parallel N`
なら N タスクずつ同時に）、各タスクの中で試行を順に回し、最後に Zero-shot pass@1・error_rate・
実行時間を集計する（`suite_summary.py`）。セッションを開く前に GPU マシンの空きを確かめ、
混んでいれば空くまで待つ。

失敗は 3 種類に分ける。推論の失敗（生成コードのエラー・Agent の例外・予算超過）は
Agent の指標に数える。インフラの失敗（GPU マシン・LLM サーバ）は同じ試行をやり直し、
使い切ったら分母から外す。設定の失敗（LLM の認証など）はスイートを止める。
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import tyro
import yaml

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:  # `suite_summary` / `trial` を、どこから実行しても読めるように
    sys.path.insert(0, str(HERE))

import envfile  # noqa: E402, F401  # remote/.env を読む（`--env-file` で別のファイル）
import suite_summary  # noqa: E402
import tracing  # noqa: E402
from suite_summary import CONFIG_KINDS, INFRA_EVENTS, LLM_INFRA_KINDS  # noqa: E402
from trial import (  # noqa: E402
    MAX_INFRA_RETRIES,
    _make_run_dir,
    _open_env,
    _resolve_agent,
    _retry_infra,
)

from capx.agent_api import (  # noqa: E402
    Budget,
    BudgetExceeded,
    CapacityFull,
    EnvStartFailed,
    EnvUnavailable,
)
from capx.bench import load_agent, run_trial  # noqa: E402
from capx.bench.args import AgentArgs  # noqa: E402
from capx.bench.errors import explain, explain_load  # noqa: E402
from capx.bench.schema import git_info  # noqa: E402

#: `max_steps` を省略したときの値（worker の既定と同じ）。
DEFAULT_MAX_STEPS = Budget().max_steps
#: スイートに書ける `max_steps` の上限。GPU マシン側の上限（backend の
#: `--worker-max-steps`。既定 20）に合わせてある。
MAX_STEPS_LIMIT = 20

#: 同時に回すタスクの数（`--parallel`）の上限。GPU マシンの枠は全員で共有する。
MAX_PARALLEL = 10
#: `--parallel` が 2 以上のとき、ほかの人のために空けておくセッションの枠の数。
RESERVED_SLOTS = 2
#: GPU マシンが空くのを待つとき、見直す間隔と、諦めるまでの時間（秒）。
SLOT_POLL_S = 15.0
SLOT_WAIT_LIMIT_S = 30 * 60.0


# -- スイートの設定 ------------------------------------------------------


class SuiteConfigError(ValueError):
    """スイートの YAML が正しくない。問題をまとめて持つ。"""


@dataclass(frozen=True)
class SuiteConfig:
    name: str
    tasks: tuple[str, ...]
    trials: int = 1
    seed: int | None = None
    max_steps: int = DEFAULT_MAX_STEPS

    def to_yaml(self) -> str:
        data = {**asdict(self), "tasks": list(self.tasks)}
        return yaml.safe_dump(data, allow_unicode=True, sort_keys=False)


_NAME = re.compile(r"^[A-Za-z0-9_-]+$")
_KEYS = ("name", "trials", "seed", "max_steps", "tasks")


def load_suite(path: str | Path) -> SuiteConfig:
    """YAML を読んで検証する。問題は 1 つずつではなく、まとめて報告する。"""
    path = Path(path)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise SuiteConfigError(f"{path}: YAML として読めない: {exc}") from exc
    if not isinstance(data, dict):
        raise SuiteConfigError(f"{path}: 中身が `キー: 値` の形になっていない")

    problems = [f"知らないキー `{k}`（使えるのは {', '.join(_KEYS)}）" for k in data if k not in _KEYS]

    name = data.get("name", path.stem)
    if not isinstance(name, str) or not _NAME.match(name):
        problems.append(f"name は英数字・`_`・`-` だけにする（今: {name!r}）")

    trials = data.get("trials", 1)
    if not _is_int(trials) or trials < 1:
        problems.append(f"trials は 1 以上の整数（今: {trials!r}）")

    seed = data.get("seed")
    if seed is not None and (not _is_int(seed) or seed < 1):
        problems.append(f"seed は 1 以上の整数。毎回ランダムにするなら書かない（今: {seed!r}）")

    max_steps = data.get("max_steps", DEFAULT_MAX_STEPS)
    if not _is_int(max_steps) or not 1 <= max_steps <= MAX_STEPS_LIMIT:
        problems.append(f"max_steps は 1〜{MAX_STEPS_LIMIT} の整数（今: {max_steps!r}）")

    tasks = data.get("tasks")
    if not isinstance(tasks, list) or not tasks or not all(isinstance(t, str) for t in tasks):
        problems.append("tasks はタスク名のリストで、1 つ以上書く")
        tasks = []
    else:
        dupes = sorted({t for t in tasks if tasks.count(t) > 1})
        if dupes:
            problems.append(f"tasks に同じタスクが 2 回以上ある: {', '.join(dupes)}")

    if problems:
        raise SuiteConfigError(f"{path} に問題がある:\n" + "\n".join(f"  - {p}" for p in problems))
    return SuiteConfig(name=name, tasks=tuple(tasks), trials=trials, seed=seed, max_steps=max_steps)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


# -- max_steps の強制 ----------------------------------------------------


class StepLimitedEnv:
    """`step()` の回数を手元で数え、`max_steps` を超える呼び出しを止める。

    GPU マシン側の上限は worker の起動時に決まり、手元からは変えられない。そこで
    スイートごとの上限をここでかける（GPU マシン側の上限より大きくしても、そちらで止まる）。超えた `step()` は GPU マシンへ送らず、
    `BudgetExceeded` を投げる。Bench（`run_trial`）がそれを予算超過として記録し、
    その時点の状態で採点する。

    手元のプロセスの中でかけている制限なので、Agent が内側の env を直接触れば
    回避できる。自分たちの評価には十分で、強制が要るならサーバ側でかける。
    """

    def __init__(self, env: Any, max_steps: int) -> None:
        self._env = env
        self._max_steps = max_steps
        self._used = 0

    def step(self, code: str, *, capture_video: bool = False):
        if self._used >= self._max_steps:
            raise BudgetExceeded("execution_budget", f"step の上限 {self._max_steps} に達した")
        self._used += 1
        return self._env.step(code, capture_video=capture_video)

    def render(self, camera: str = "main") -> bytes:
        return self._env.render(camera)

    def reset_for_trial(self, trial: int, seed: int | None = None):
        self._used = 0
        return self._env.reset_for_trial(trial, seed)

    def __getattr__(self, name: str) -> Any:
        # Bench が使う evaluate / recorded_steps / rendered_images / close などはそのまま通す
        return getattr(self._env, name)


# -- 実行 ----------------------------------------------------------------


class SuiteAbortError(RuntimeError):
    """設定の失敗。続けても全部失敗するので、スイートを止める。"""


class _LlmInfraError(Exception):
    """LLM サーバ側の失敗。結果（`failure.kind`）として返ってきたものを、やり直すために例外にする。"""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(f"[{kind}] {message}")
        self.kind = kind
        self.message = message


#: やり直しを使い切ったら、その試行だけ捨てずにスイートを止める失敗。LLM に繋がらない
#: 状態が続くなら、残りの試行も全部同じ理由で捨てることになる。
STOP_WHEN_EXHAUSTED = ("llm_unreachable",)


@dataclass
class Runner:
    """1 回の実行に必要なものをまとめて持つ。"""

    suite: SuiteConfig
    run_dir: Path
    agent_path: Path
    agent_sha256: str
    model: str
    record_video: bool
    server_url: str | None
    endpoint: str | None
    sleep: Any = time.sleep
    parallel: int = 1
    _stop: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _done: int = field(default=0, init=False, repr=False)

    def run(self) -> None:
        if self.parallel <= 1:
            for task in self.suite.tasks:
                self.run_task(task)
            return
        # タスク単位で並べる。同じタスクの試行は 1 つのセッションで順に回す
        with ThreadPoolExecutor(max_workers=self.parallel, thread_name_prefix="suite") as pool:
            futures = [pool.submit(self.run_task, task) for task in self.suite.tasks]
            try:
                for future in as_completed(futures):
                    future.result()
            except BaseException:
                # 回し始めた試行は途中で止められないので、終わるのを待つ（with を抜けるとき）
                self._stop.set()
                for future in futures:
                    future.cancel()
                print("止める。実行中の試行が終わるのを待つ")
                raise

    def run_task(self, task: str) -> None:
        """1 タスクの試行を順に回す。セッションは試行の間で使い回す。"""
        if self._stop.is_set():
            return
        (self.run_dir / task).mkdir(parents=True, exist_ok=True)
        holder: dict[str, Any] = {"env": None}
        try:
            for trial in range(1, self.suite.trials + 1):
                if self._stop.is_set():
                    return
                line = self.run_one(holder, task, trial)
                with self._lock:
                    self._done += 1
                    planned = len(self.suite.tasks) * self.suite.trials
                    print(f"[{self._done}/{planned}] {task} trial {trial}: {line}")
        except BaseException:
            # ほかのスレッドが次の試行・タスクを始めないよう、ここで止める合図を出す
            self._stop.set()
            raise
        finally:
            if holder["env"] is not None:
                holder["env"].close()

    def run_one(self, holder: dict[str, Any], task: str, trial: int) -> str:
        """1 試行を、インフラの失敗ならやり直しながら回す。進捗の 1 行を返す。"""
        task_dir = self.run_dir / task
        seed = self.suite.seed if self.suite.seed is not None else random.SystemRandom().randint(1, 2**31 - 1)
        retried: list[str] = []

        def on_failure(exc: Exception, retries: int, gave_up: bool) -> None:
            kind, category, message = _classify(exc)
            if category == "gpu" and holder["env"] is not None:
                # 中断した後の環境は状態が分からないので、使い回さず作り直す
                holder["env"].close()
                holder["env"] = None
            _append_event(task_dir, {
                "trial": trial,
                "seed": seed,
                "attempt": retries + 1,
                "category": category,
                "kind": kind,
                "message": message,
                "gave_up": gave_up,
                "at": _now(),
            })
            if not gave_up:
                retried.append(kind)

        try:
            summary = _retry_infra(
                lambda retries: self._attempt(holder, task, trial, seed, retries),
                retryable=(EnvUnavailable, _LlmInfraError),
                on_failure=on_failure,
                action="同じ試行をやり直す",
                describe=lambda exc: "[{}] {}".format(*_classify(exc)[::2]),
                sleep=self.sleep,
            )
        except (EnvUnavailable, _LlmInfraError) as exc:
            kind = _classify(exc)[0]
            if kind in STOP_WHEN_EXHAUSTED:
                raise SuiteAbortError(f"{exc}（{MAX_INFRA_RETRIES} 回やり直しても直らない）") from exc
            return f"infra_failed（{kind}）"
        return _progress(summary, retried)

    def _attempt(self, holder, task, trial, seed, retries):
        if holder["env"] is None:
            holder["env"] = self._connect(task)
        with tracing.trial_trace(
            session=tracing.session_id(self.run_dir, self.model, task),
            task=task,
            trial=trial,
            seed=seed,
            attempt=retries + 1,
            model=self.model,
            agent_path=self.agent_path,
            tags=["suite", f"suite:{self.suite.name}"],
            infra_errors=(EnvUnavailable, _LlmInfraError),
        ) as trace:
            summary = self._run_trial(trace, holder, task, trial, seed, retries)
            trace.finish(summary)
            return summary

    def _run_trial(self, trace, holder, task, trial, seed, retries):
        # 外側に TracedEnv を置き、上限で止めた step（BudgetExceeded）も記録に残す
        summary = run_trial(
            trace.wrap(StepLimitedEnv(holder["env"], self.suite.max_steps)),
            self._load_agent(),
            trial,
            {"output_dir": str(self.run_dir / task), "record_video": self.record_video},
            Budget(max_steps=self.suite.max_steps),
            seed=seed,
            meta={"agent_spec": str(self.agent_path), "model": self.model, "endpoint_task": task},
            infrastructure_retries=retries,
        )
        failure = (summary.result.failure or {}) if summary.result else {}
        kind = failure.get("kind")
        if kind in CONFIG_KINDS:
            _discard(summary, self.run_dir / task)
            raise SuiteAbortError(failure.get("message") or kind)
        if kind in LLM_INFRA_KINDS:
            _discard(summary, self.run_dir / task)
            raise _LlmInfraError(kind, failure.get("message", ""))
        return summary

    def _connect(self, task: str):
        """GPU マシンが空くのを待ってから、セッションを開く。

        空きは `GET /sessions` で確かめる。確かめた直後にほかの人が埋めて 429 が返ったときも、
        待ってからもう一度開く。どちらも、インフラの失敗のやり直しの回数には数えない。
        `SLOT_WAIT_LIMIT_S` 待っても空かなければ `CapacityFull` を投げる（そこからは
        インフラの失敗として扱う）。
        """
        waited = 0.0
        while True:
            busy = self._busy(task)
            if busy is None:
                try:
                    return self._open(task)
                except CapacityFull as exc:
                    busy = str(exc)
            if waited >= SLOT_WAIT_LIMIT_S:
                raise CapacityFull(f"{waited / 60:.0f} 分待っても GPU マシンが空かない: {busy}")
            if not waited:
                print(f"{task}: GPU マシンが混んでいる（{busy}）。空くのを待つ")
            self.sleep(SLOT_POLL_S)
            waited += SLOT_POLL_S

    def _busy(self, task: str) -> str | None:
        """新しくセッションを開けるか。開けるなら None、開けないなら理由。"""
        if self.endpoint:
            return None
        try:
            now = _occupancy(self.server_url)
        except Exception:
            return None  # 見られない（古い backend など）ときは、開いてみて 429 で判断する
        reserve = RESERVED_SLOTS if self.parallel > 1 else 0
        free = now["capacity"] - now["active"]
        if free <= reserve:
            return f"空き {free} 枠。ほかの人のために {reserve} 枠は空けておく" if reserve else "空き 0 枠"
        if "memory_budget_gb" in now and "memory_used_gb" in now:
            free_gb = now["memory_budget_gb"] - now["memory_used_gb"]
            need_gb = _session_memory_gb(task) * (1 + reserve)
            if free_gb < need_gb:
                return f"メモリの空き {free_gb:.0f}GB（{need_gb:.0f}GB 要る）"
        return None

    def _open(self, task: str):
        """セッションを開く。直らない失敗は止め、一時的な失敗はやり直せる形で投げる。"""
        import requests

        try:
            return _open_env(self.endpoint, self.server_url, task, self.record_video)
        except EnvStartFailed as exc:
            code = exc.status_code
            if code is not None and 400 <= code < 500 and code != 429:
                # 認証・タスク名の誤りなど。やり直しても直らない
                raise SuiteAbortError(str(explain(exc))) from exc
            raise
        except requests.RequestException as exc:
            # `_create_session` は通信の失敗を包まずに投げる。GPU マシン側の一時的な失敗として扱う
            raise EnvUnavailable(f"GPU マシンとの通信に失敗した: {exc!r}") from exc

    def _load_agent(self):
        # Zero-shot を保つため、試行ごとに Agent を読み込み直す（前の試行の状態を持ち越させない）。
        # 実行中にファイルが書き換わると、別の Agent の結果が混ざるので止める。
        try:
            changed = _sha256(self.agent_path) != self.agent_sha256
        except OSError as exc:
            raise SuiteAbortError(f"Agent ファイルを読めない: {self.agent_path}（{exc}）") from exc
        if changed:
            raise SuiteAbortError(
                f"Agent ファイルが実行中に書き換わった: {self.agent_path}\n"
                "  違う Agent の結果が混ざるので止めた。直してから実行し直す"
            )
        try:
            return load_agent(str(self.agent_path), ctx=AgentArgs(model=self.model).to_context())
        except Exception as exc:
            raise SuiteAbortError(str(explain_load(exc, str(self.agent_path)))) from exc


def _classify(exc: Exception) -> tuple[str, str, str]:
    """インフラの失敗を `(kind, gpu / llm, メッセージ)` にする。"""
    if isinstance(exc, _LlmInfraError):
        return exc.kind, "llm", exc.message
    explained = explain(exc)
    return explained.kind, "gpu", explained.message


def _discard(summary, task_dir: Path) -> None:
    """集計に入れない試行のフォルダを `_discarded/` に移す。調べられるように消さない。

    名前には移した時刻を付ける。同じ試行が何度捨てられても、前のものを上書きしない。
    """
    if not summary.code_path:
        return
    trial_dir = Path(summary.code_path).parent
    base = task_dir / "_discarded" / f"{trial_dir.name}_{time.strftime('%Y%m%d_%H%M%S')}"
    base.parent.mkdir(exist_ok=True)
    target, n = base, 2
    while target.exists():
        target = base.with_name(f"{base.name}_{n}")
        n += 1
    shutil.move(str(trial_dir), str(target))


def _progress(summary, retries: list[str]) -> str:
    result = summary.result
    parts = [
        "completed" if summary.task_completed else "not completed",
        f"{summary.num_code_blocks} steps",
        f"{result.wall_clock_s:.1f}s" if result else "",
    ]
    if result and result.failure:
        parts.append(f"[{result.failure['kind']}]")
    if retries:
        parts.append(f"(infra retry {len(retries)}: {', '.join(retries)})")
    return ", ".join(p for p in parts if p)


def _append_event(task_dir: Path, event: dict[str, Any]) -> None:
    with (task_dir / INFRA_EVENTS).open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _session_memory_gb(task: str) -> float:
    """1 セッションが GPU マシンで借りるメモリ（GB）。backend の既定（`RUNTIME_MEMORY`）と同じ。"""
    return 5.0 if task.startswith("libero_") else 4.0


def _occupancy(server_url: str) -> dict[str, Any]:
    """GPU マシンの今の混み具合（`active` / `capacity` / `memory_used_gb` / `memory_budget_gb`）。"""
    import requests

    from capx.remote_env.client import _auth_headers

    reply = requests.get(f"{server_url.rstrip('/')}/sessions", headers=_auth_headers(), timeout=10)
    reply.raise_for_status()
    return reply.json()


def _available_tasks(server_url: str) -> list[str]:
    import requests

    from capx.remote_env.client import _auth_headers

    reply = requests.get(f"{server_url.rstrip('/')}/tasks", headers=_auth_headers(), timeout=10)
    reply.raise_for_status()
    return list(reply.json().get("tasks", []))


def _check_target(suite: SuiteConfig, server_url: str | None, endpoint: str | None) -> None:
    """接続先でスイートを回せるか。回し始める前に確かめる。"""
    if endpoint:
        if len(suite.tasks) > 1:
            # 直接つないだ worker は 1 タスク分しか無く、閉じると止まる
            sys.exit("`--endpoint` で回せるのはタスクが 1 つのスイートだけ")
        return
    try:
        available = set(_available_tasks(server_url))
    except Exception as exc:
        sys.exit(f"GPU マシンからタスクの一覧を取れない（{exc}）。`doctor.py` で原因を調べる\n  接続先: {server_url}")
    missing = [t for t in suite.tasks if t not in available]
    if missing:
        sys.exit(
            f"GPU マシンに無いタスクがある: {', '.join(missing)}\n"
            "  選べるタスクは `uv run doctor.py` で見られる"
        )


# -- 入り口 --------------------------------------------------------------


def main(
    suite: str,
    /,
    *,
    agent: str | None = None,
    model: str | None = None,
    output_dir: str | None = None,
    record_video: bool = True,
    parallel: int = 1,
    trace: bool = True,
    trace_step_videos: bool = False,
    endpoint: str | None = None,
) -> None:
    """スイートの全タスクを回し、最後に集計する。

    Args:
        suite: スイートの YAML。`remote/` からの相対パスでも、実行した場所からでもよい。
        agent: Agent ファイル。省略すると `agents/example_agent.py`。
        model: LLM のモデル名。省略すると環境変数 OPENAI_BASE_MODEL。
        output_dir: 結果を置く場所。省略すると `remote/outputs/`。その下の
            `<モデル名>/suite_<名前>/<日時>/` に、タスクごと・試行ごとのフォルダを作る。
        record_video: `--no-record-video` で動画を保存しない。
        parallel: 同時に回すタスクの数（1〜10）。GPU マシンの空きを見て、混んでいれば待つ。
            並べると GPU・LLM サーバを取り合うので、実行時間は 1 本ずつより長くなる。
        trace: `--no-trace` で Langfuse に送らない。キー（LANGFUSE_PUBLIC_KEY /
            LANGFUSE_SECRET_KEY）が無ければ、付けなくても送らない。
        trace_step_videos: `env.step` の記録に、その step の動画も付ける。Langfuse の容量が
            試行あたり約 2 倍になるので、失敗した試行を詳しく見たいときだけ使う。
        endpoint: worker に直接つなぐ（`tcp://host:19500`）。タスクが 1 つのスイートだけ。
            通常は使わない。
    """
    # ファイルや `tee` に流したときも、進捗を 1 行ずつすぐに出す（既定ではためてから書く）
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    server_url = os.environ.get("CAPX_ENV_SERVER_URL")
    if not endpoint and not server_url:
        sys.exit(
            "GPU マシンの場所が分からない。環境変数 CAPX_ENV_SERVER_URL を設定する"
            "（remote/.env.example をコピーして .env にし、値を入れる）"
        )

    if not 1 <= parallel <= MAX_PARALLEL:
        sys.exit(f"--parallel は 1〜{MAX_PARALLEL}（GPU マシンの枠は全員で共有する）")
    runner = _prepare(suite, agent, model, output_dir, record_video, parallel, server_url, endpoint)

    print(f"保存先: {runner.run_dir}")
    if tracing.setup(trace, step_videos=trace_step_videos):
        print(f"トレース: Langfuse の session {tracing.session_id(runner.run_dir, runner.model, '')}")
    stopped = None
    try:
        runner.run()
    except SuiteAbortError as exc:
        stopped = f"スイートを止めた: {exc}\n  直してから、もう一度実行する"
    except KeyboardInterrupt:
        stopped = "中断した"
    finally:
        tracing.flush()
        _update_run(runner.run_dir, finished_at=_now())
        print()
        print(suite_summary.render(suite_summary.write_summary(runner.run_dir)))
    if stopped:
        sys.exit(stopped)


def _prepare(suite_path, agent, model, output_dir, record_video, parallel, server_url, endpoint) -> Runner:
    path = _resolve_suite(suite_path)
    try:
        config = load_suite(path)
    except SuiteConfigError as exc:
        sys.exit(str(exc))
    _check_target(config, server_url, endpoint)
    agent_path = _resolve_agent(agent or "agents/example_agent.py")
    model = model if model is not None else os.environ.get("OPENAI_BASE_MODEL", "")
    _try_load(agent_path, model)
    parallel = min(parallel, len(config.tasks))

    sha = _sha256(agent_path)
    run_dir = _make_run_dir(output_dir, model, f"suite_{config.name}")
    (run_dir / "suite.yaml").write_text(config.to_yaml(), encoding="utf-8")
    shutil.copy2(agent_path, run_dir / "agent.py")
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "agent": {"spec": str(agent_path), "sha256": sha, "model": model},
                "record_video": record_video,
                "parallel": parallel,
                "git": git_info(),
                "started_at": _now(),
                "finished_at": None,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return Runner(
        config, run_dir, agent_path, sha, model, record_video, server_url, endpoint, parallel=parallel
    )


def _try_load(agent_path: Path, model: str) -> None:
    """回し始める前に 1 回読み込んで、構文エラーなどをここで出す。"""
    print(f"Agent : {agent_path}")
    try:
        load_agent(str(agent_path), ctx=AgentArgs(model=model).to_context())
    except Exception as exc:
        sys.exit(str(explain_load(exc, str(agent_path))))


def _resolve_suite(spec: str) -> Path:
    for candidate in (Path(spec), HERE / spec):
        if candidate.exists():
            return candidate.resolve()
    sys.exit(f"スイートの YAML が見つからない: {spec}\n  探した場所: {Path(spec).resolve()} と {HERE / spec}")


def _update_run(run_dir: Path, **fields: Any) -> None:
    path = run_dir / "run.json"
    run = json.loads(path.read_text(encoding="utf-8"))
    run.update(fields)
    path.write_text(json.dumps(run, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    tyro.cli(main)
