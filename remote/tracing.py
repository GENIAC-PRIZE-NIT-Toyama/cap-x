"""trial ごとの挙動を Langfuse に送る。`trial.py` と `suite.py` が使う。

1 attempt（インフラ障害でやり直したら、やり直すたびに別）が 1 トレースになる。

    trace（root span、名前はタスク名）  input: task・instruction・seed / output: 結果と動画
    ├─ generation  LLM の呼び出し（OpenAI SDK を自動で記録する）
    ├─ tool        env.step（input: 送ったコード / output: ok・stdout・stderr）
    └─ ...         呼ばれた順に並ぶ
    score: reward / task_completed / exec_ok / failure_kind

Snapshot を有効にした env なら、trace の input と metadata に reset 直後の Snapshot ID
（`initial_snapshot_id`）を、各 env.step の metadata に `agent_step` / `snapshot_before` /
`snapshot_after` を付ける。Snapshot から再開した実行（Fork）は `fork_trace` で新しい trace に
し、元の trace / observation / Snapshot ID を metadata（`parent_trace_id` など）に残す。

Agent の作者は何もしなくてよい。`langfuse.openai` を読み込むと OpenAI SDK のメソッドが
クラスごと書き換わるので、同梱の `query_model` も、Agent が自分で作った `OpenAI()` も
記録される。OpenAI SDK 以外（anthropic など）の呼び出しは記録されない。

`LANGFUSE_PUBLIC_KEY` と `LANGFUSE_SECRET_KEY` が無ければ何もしない（langfuse を
import もしない）。誰の実行かは `CAPX_USER` で区別する。
"""

from __future__ import annotations

import hashlib
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from suite_summary import CONFIG_KINDS, LLM_INFRA_KINDS

from capx.agent_api import BudgetExceeded

#: 有効なときの Langfuse クライアント。無効なら None で、どの関数も何もしない。
_client: Any = None
#: step ごとの動画（ターン動画）も送る（`--trace-step-videos`）。容量が試行あたり約 2 倍になる。
_step_videos = False

#: Langfuse に送る動画の大きさの上限。超えたら送らない（手元には残っている）。
MAX_VIDEO_BYTES = 20 * 1024 * 1024

#: インフラ障害で終わった attempt に付けるタグ。score が付かないので、これで探す。
INFRA_FAILED_TAG = "infra_failed"


def setup(enabled: bool = True, *, step_videos: bool = False) -> bool:
    """キーがあれば記録を始める。始めたら True。

    Args:
        step_videos: `env.step` の記録に、その step の動画も付ける。動画を受け取って
            いない（`--no-record-video` / `--no-save`）ときは付かない。
    """
    global _client, _step_videos
    if not enabled or not (os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")):
        return False
    import langfuse.openai  # noqa: F401  読み込むだけで OpenAI SDK の呼び出しが記録される
    from langfuse import get_client

    _client = get_client()
    _step_videos = step_videos
    return True


def flush() -> None:
    """送り残しを送る。最後の trial が届かないことがあるので、終わるときに必ず呼ぶ。"""
    if _client is not None:
        _client.flush()


def session_id(run_dir: Path | None, model: str, task: str) -> str:
    """`<モデル名>/<タスク名 or suite_名前>/<日時>`。保存先のフォルダと同じにして、行き来しやすくする。"""
    if run_dir is not None:
        return "/".join(run_dir.parts[-3:])
    return f"{model.replace('/', '_') or 'model_unset'}/{task}/{time.strftime('%Y%m%d_%H%M%S')}"


@contextmanager
def trial_trace(
    *,
    session: str,
    task: str,
    trial: int,
    seed: int | None,
    attempt: int,
    model: str,
    agent_path: str | Path,
    tags: list[str],
    infra_errors: tuple[type[BaseException], ...] = (),
    metadata: dict[str, Any] | None = None,
) -> Iterator[TrialTrace]:
    """1 attempt を 1 トレースで包む。

    例外はそのまま投げる。`infra_errors` の例外なら `infra_failed` タグを付ける。
    `metadata` は trace の metadata に足す（値は 200 文字までの文字列になる）。
    """
    if _client is None:
        yield TrialTrace(None)
        return
    from langfuse import propagate_attributes

    agent_path = Path(agent_path)
    # 値は US-ASCII・200 文字以内でないと捨てられる。instruction など長いものは input に入れる
    attributes = propagate_attributes(
        user_id=_user_id(),
        session_id=_ascii(session),
        trace_name=_ascii(task),
        version=_digest(agent_path),
        tags=[_ascii(tag) for tag in tags],
        metadata={
            "task": task,
            "trial": trial,
            "seed": seed,
            "attempt": attempt,
            "model": model,
            "agent": agent_path.name,
            **{k: v for k, v in (metadata or {}).items() if v is not None},
        },
    )
    with _client.start_as_current_observation(as_type="span", name=task) as span, attributes:
        try:
            yield TrialTrace(span)
        except BaseException as exc:
            if isinstance(exc, infra_errors):
                with propagate_attributes(tags=[INFRA_FAILED_TAG]):
                    pass
            span.update(level="ERROR", status_message=f"{type(exc).__name__}: {exc}"[:1000])
            raise


@contextmanager
def fork_trace(
    *,
    session: str,
    task: str,
    snapshot_id: str,
    fork_type: str,
    model: str,
    agent_path: str | Path,
    parent_trace_id: str | None = None,
    parent_observation_id: str | None = None,
    trial: int | None = None,
    seed: int | None = None,
    tags: list[str] | None = None,
    infra_errors: tuple[type[BaseException], ...] = (),
) -> Iterator[TrialTrace]:
    """Snapshot から再開した実行（Fork）を、新しい 1 トレースで包む。

    元の trace とは metadata でつなぐ（Langfuse の検索で `fork_snapshot_id` などを引ける）。
    `parent_trace_id` / `parent_observation_id` を省略しても、`wrap` した env が Snapshot の
    記録から持っていれば（`resume_info["meta"]`）、そちらを使う。

    Args:
        fork_type: 分岐の種類。`human_edit`（人が直したコード）/ `best_of_n` / `retry` /
            `resume`（同じ Agent で続けるだけ）など。自由な文字列。
    """
    fork = {
        "fork_snapshot_id": snapshot_id,
        "fork_type": fork_type,
        "parent_trace_id": parent_trace_id,
        "parent_observation_id": parent_observation_id,
    }
    with trial_trace(
        session=session,
        task=task,
        trial=trial if trial is not None else 0,
        seed=seed,
        attempt=1,
        model=model,
        agent_path=agent_path,
        tags=["fork", f"fork:{fork_type}", *(tags or [])],
        infra_errors=infra_errors,
        metadata=fork,
    ) as trace:
        trace._fork = fork
        yield trace


class TrialTrace:
    """`trial_trace` が返す。無効なときは全部何もしない。"""

    def __init__(self, span: Any) -> None:
        self._span = span
        self._fork: dict[str, Any] | None = None

    @property
    def trace_id(self) -> str | None:
        return None if self._span is None else self._span.trace_id

    @property
    def url(self) -> str | None:
        if self._span is None:
            return None
        try:
            return _client.get_trace_url(trace_id=self._span.trace_id)
        except Exception:
            return None

    def wrap(self, env: Any) -> Any:
        """`step` を記録する env にする。無効なら env をそのまま返す。

        Snapshot から再開した env なら、再開した時点の情報を trace の input に入れる。
        """
        if self._span is None:
            return env
        if getattr(env, "resume_from", None):
            self.set_resume(env)
        return TracedEnv(env, self)

    def set_task(self, task: Any, trial: int, snapshot_id: str | None = None) -> None:
        if self._span is None:
            return
        # dataset の item にしやすいよう、再現に要るものを input にまとめる
        data = {
            "task_id": task.task_id,
            "instruction": task.instruction,
            "seed": task.seed,
            "trial": trial,
        }
        if snapshot_id:
            data["initial_snapshot_id"] = snapshot_id
            self._span.update(metadata={"initial_snapshot_id": snapshot_id})
        self._span.update(input=data)

    def set_resume(self, env: Any) -> None:
        """再開した env の情報（どの Snapshot の何 Step 目からか、元の trace）を input にする。"""
        if self._span is None:
            return
        info = getattr(env, "resume_info", None) or {}
        meta = info.get("meta") or {}
        fork = dict(self._fork or {})
        # 元の trace / observation は、Snapshot を保存したときの記録から補う
        fork.setdefault("fork_snapshot_id", env.resume_from)
        if not fork.get("parent_trace_id"):
            fork["parent_trace_id"] = meta.get("trace_id")
        if not fork.get("parent_observation_id"):
            fork["parent_observation_id"] = meta.get("observation_id")
        task = getattr(env, "task", None)
        self._span.update(
            input={
                "task_id": getattr(task, "task_id", None),
                "instruction": getattr(task, "instruction", None),
                "seed": getattr(task, "seed", None),
                "resume_from": env.resume_from,
                "agent_step": info.get("agent_step"),
                **fork,
            },
            metadata={k: v for k, v in fork.items() if v is not None},
        )

    def finish(self, summary: Any) -> None:
        """score と結果を付ける。動画が保存されていれば一緒に送る。

        記録は評価のおまけなので、ここで失敗しても試行は止めない（警告だけ出す）。
        """
        if self._span is None:
            return
        try:
            self._finish(summary)
        except Exception as exc:
            print(f"Langfuse に結果を付けられなかった（試行は続ける）: {type(exc).__name__}: {exc}")

    def _finish(self, summary: Any) -> None:
        failure = (summary.result.failure or {}) if summary.result is not None else {}
        if failure.get("kind") in LLM_INFRA_KINDS + CONFIG_KINDS:
            # Agent の責任ではない失敗。`suite.py` が集計から外すのと同じく、score を付けない
            from langfuse import propagate_attributes

            with propagate_attributes(tags=[INFRA_FAILED_TAG]):
                self._span.update(output={"failure": failure}, level="ERROR", status_message=failure.get("message"))
            return
        self._span.score_trace(name="reward", value=float(summary.reward), data_type="NUMERIC")
        if summary.task_completed is not None:
            self._span.score_trace(name="task_completed", value=int(bool(summary.task_completed)), data_type="BOOLEAN")
        self._span.score_trace(name="exec_ok", value=int(bool(summary.success)), data_type="BOOLEAN")
        if failure.get("kind"):
            self._span.score_trace(name="failure_kind", value=failure["kind"], data_type="CATEGORICAL")

        output: dict[str, Any] = {
            "reward": summary.reward,
            "task_completed": summary.task_completed,
            "exec_ok": summary.success,
            "steps": summary.num_code_blocks,
        }
        if failure:
            output["failure"] = failure
        video = _video(summary)
        if video is not None:
            output["video"] = video
        self._span.update(output=output)


class TracedEnv:
    """`step` を tool observation として記録する。ほかはそのまま通す。

    Agent が呼ぶ `step` と LLM の呼び出しが同じコンテキストで動くので、
    generation → step → generation… と呼んだ順に並ぶ。
    """

    def __init__(self, env: Any, trace: TrialTrace) -> None:
        self._env = env
        self._trace = trace

    def step(self, code: str, *, capture_video: bool = False):
        before = getattr(self._env, "last_snapshot_id", None)
        with _client.start_as_current_observation(
            as_type="tool", name="env.step", input={"code": _code_block(code, "python")}
        ) as obs:
            # Snapshot の記録に、この env.step の trace / observation を残す（Fork から辿れる）
            context = getattr(self._env, "snapshot_context", None)
            if isinstance(context, dict):
                context.update(trace_id=obs.trace_id, observation_id=obs.id)
            try:
                result = self._env.step(code, capture_video=capture_video)
            except BudgetExceeded as exc:  # Agent の予算超過。送らずに止めた
                obs.update(level="WARNING", status_message=str(exc))
                raise
            except Exception as exc:
                obs.update(level="ERROR", status_message=f"{type(exc).__name__}: {exc}"[:1000])
                raise
            output = {"ok": result.ok, "stdout": _code_block(result.stdout), "stderr": _code_block(result.stderr)}
            if _step_videos and result.video:
                from langfuse.media import LangfuseMedia

                output["video"] = LangfuseMedia(
                    content_type=result.video_media_type or "video/mp4", content_bytes=result.video
                )
            obs.update(
                output=output,
                level="DEFAULT" if result.ok else "WARNING",
                metadata=self._snapshot_metadata(before, result),
            )
            return result

    def _snapshot_metadata(self, before: str | None, result: Any) -> dict[str, Any] | None:
        if not getattr(self._env, "snapshot_enabled", False) and not getattr(result, "snapshot_id", None):
            return None
        meta = {
            "agent_step": getattr(self._env, "agent_step", None),
            "snapshot_before": before,
            "snapshot_after": result.snapshot_id,
        }
        error = getattr(self._env, "last_snapshot_error", None)
        if result.snapshot_id is None and error:
            meta["snapshot_error"] = error[:2000]
        return meta

    def render(self, camera: str = "main") -> bytes:
        return self._env.render(camera)

    def reset_for_trial(self, trial: int, seed: int | None = None):
        context = getattr(self._env, "snapshot_context", None)
        if isinstance(context, dict) and self._trace._span is not None:
            context.update(trace_id=self._trace._span.trace_id, observation_id=self._trace._span.id)
        task = self._env.reset_for_trial(trial, seed)
        self._trace.set_task(task, trial, snapshot_id=getattr(self._env, "initial_snapshot_id", None))
        return task

    def __getattr__(self, name: str) -> Any:
        # Bench が使う evaluate / recorded_steps / close などはそのまま通す
        return getattr(self._env, name)


def _code_block(text: str, lang: str = "") -> str:
    """Markdown のコードブロックに入れる。空ならそのまま。

    Langfuse の UI は文字列を Markdown として表示するので、そのままだと Python の `#` が
    見出しに、`*` が強調になる。Langfuse には見るための形で送り、生のデータは手元の
    `steps/` と `result.json` を正とする。
    """
    if not text:
        return text
    fence = "```"
    while fence in text:  # 中に ``` があっても閉じないよう、囲みを長くする
        fence += "`"
    body = text if text.endswith("\n") else text + "\n"
    return f"{fence}{lang}\n{body}{fence}"


def _video(summary: Any) -> Any:
    """trial のフォルダに保存された動画。無ければ None。

    つないだ `combined.mp4` を送る。ターンが 1 つのときや ffmpeg が無いときは作られない
    ので、そのときはターンごとの動画を並べて送る。
    """
    if not summary.code_path:
        return None
    video_dir = Path(summary.code_path).parent / "videos"
    combined = video_dir / "combined.mp4"
    paths = [combined] if combined.is_file() else sorted(video_dir.glob("turn_*.mp4"))
    if not paths or sum(p.stat().st_size for p in paths) > MAX_VIDEO_BYTES:
        return None
    from langfuse.media import LangfuseMedia

    media = [LangfuseMedia(content_type="video/mp4", content_bytes=p.read_bytes()) for p in paths]
    return media[0] if len(media) == 1 else media


_warned_user = False


def _user_id() -> str | None:
    global _warned_user
    user = os.environ.get("CAPX_USER", "").strip()
    if not user:
        return None
    if not user.isascii():
        if not _warned_user:
            print(f"CAPX_USER はローマ字で書く（Langfuse が受け付けない）: {user!r}。user を付けずに送る")
            _warned_user = True
        return None
    return user[:200]


def _ascii(text: str) -> str:
    return "".join(c if c.isascii() else "_" for c in text)[:200]


def _digest(path: Path) -> str | None:
    """Agent ファイルの sha256 の先頭 8 文字。同じ Agent の実行どうしを比べるのに使う。"""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:8]
    except OSError:
        return None
