"""`CodeExecutionEnvBase` を `AgentEnv` の形に包む。

Agent からは `step` と `render` しか見えない。reward と task_completed は
Bench 専用の `evaluate()` からしか取れないので、Agent 側のコードに採点結果が
流れ込む経路がない。

worker コンテナの中身としてもこれを使う。`RemoteAgentEnv` は ZMQ 越しに
この同じクラスを呼ぶので、Local と Remote で実装が二重化しない。
"""

from __future__ import annotations

import io
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from capx.agent_api import (
    Budget,
    BudgetExceeded,
    StepResult,
    TaskSpec,
    TruncationReason,
)

if TYPE_CHECKING:
    from capx.envs.tasks.base import CodeExecutionEnvBase


@dataclass
class RecordedStep:
    """Bench が成果物を書くために控えておく 1 ステップ分。

    Agent の自己申告ではなく、`step()` を通ったものをそのまま残す。
    """

    code: str
    ok: bool
    stdout: str
    stderr: str
    execution_time_s: float
    frame_range: tuple[int, int] = (0, 0)
    # 採点まわり。Local は worker と同じプロセスなので控えられるが、Remote の
    # クライアントは Agent と同じく採点を知らない（`evaluate()` でだけ取る）。
    reward: float | None = None
    task_completed: bool | None = None
    terminated: bool = False
    truncated: bool = False
    #: このステップのターン動画（mp4）。Remote で `record_video` のときだけ入る。
    video: bytes | None = None
    #: この Step が終わった時点の Snapshot ID。Snapshot を有効にした Remote でだけ入る。
    snapshot_id: str | None = None


@dataclass
class TrialOutcome:
    """`evaluate()` が返すもの。Bench だけが受け取る。"""

    reward: float
    task_completed: bool | None
    terminated: bool
    truncated: bool
    sandbox_rc: int
    steps: list[RecordedStep] = field(default_factory=list)


class LocalAgentEnv:
    """同一プロセスの `CodeExecutionEnvBase` を `AgentEnv` として見せる。

    Agent に渡すのは `step` / `render` / `close` だけ。`reset_for_trial()` と
    `evaluate()` は Bench が呼ぶもので、`AgentEnv` Protocol には含めていない
    ——Agent が型として受け取っても呼べない。

    予算はここで強制する。Agent が `budget` を無視する前提で設計している。
    """

    def __init__(
        self,
        env: CodeExecutionEnvBase,
        budget: Budget | None = None,
        *,
        record_video: bool = False,
        wrist_camera: bool = False,
        task_id: str | None = None,
        frame_budget_bytes: int | None = None,
    ) -> None:
        self._env = env
        self._task_id = task_id
        self._budget = budget or Budget()
        self._record_video = record_video
        self._wrist_camera = wrist_camera
        self._frame_budget_bytes = frame_budget_bytes

        self._steps: list[RecordedStep] = []
        self._execution_time_ns = 0
        self._last_info: dict[str, Any] = {"sandbox_rc": -1, "stdout": "", "stderr": ""}
        self._last_reward = 0.0
        self._last_terminated = False
        self._last_truncated = False
        self._closed = False
        # trial の開始時刻。`trial_wall_clock_s`（LLM 待ちも含む全体の上限）を測る。
        self._trial_started = time.monotonic()
        # Snapshot に残す、今の trial の番号と seed と TaskSpec。
        self._trial: int | None = None
        self._seed: int | None = None
        self._task: TaskSpec | None = None
        self._reset_rngs: dict[str, Any] | None = None
        # 復元のとき、次の reset の直前に戻す乱数（`restore` が入れる）
        self._restore_rngs: dict[str, Any] | None = None

    # -- Bench 専用 --------------------------------------------------------

    def reset_for_trial(self, trial: int, seed: int | None = None) -> TaskSpec:
        """環境を初期化し、Agent に渡す `TaskSpec` を作る。

        seed の既定は trial 番号（現行の `env.reset(seed=trial)` と同じ）。
        なお現状 seed は robosuite まで届いていないため、初期配置は固定
        されない（docs/RemoteDevelopment.md の Phase 0B 節）。
        """
        seed = trial if seed is None else seed
        # reset の直前の乱数。Snapshot から戻すとき、同じ乱数で reset して同じモデルを作る
        # （robosuite / LIBERO は reset のたびに、乱数で物体の大きさや位置を決めてモデルを作り直す）
        self._reset_rngs = self._capture_rngs()
        obs, info = self._env.reset(options={"trial": trial}, seed=seed)
        self._trial_started = time.monotonic()

        if self._record_video and hasattr(self._env, "enable_video_capture"):
            low = getattr(self._env, "low_level_env", None)
            if self._frame_budget_bytes is not None and hasattr(low, "set_frame_budget"):
                low.set_frame_budget(self._frame_budget_bytes)
            self._env.enable_video_capture(
                True, clear=True, wrist_camera=self._wrist_camera
            )

        self._steps.clear()
        self._execution_time_ns = 0
        self._last_info = {"sandbox_rc": -1, "stdout": "", "stderr": ""}
        self._last_reward = 0.0
        self._last_terminated = self._last_truncated = False
        self._trial, self._seed = trial, seed

        self._task = TaskSpec(
            task_id=self._task_id or type(self._env).__name__,
            seed=seed,
            instruction=info.get("task_prompt") or "",
            api_docs="\n\n".join(
                api.combined_doc() for api in getattr(self._env, "_apis", {}).values()
            ),
            default_prompt=obs.get("full_prompt", []),
            cameras=self._camera_names(),
            reference_code=None,
        )
        return self._task

    # -- Snapshot（worker が呼ぶ） -------------------------------------------

    def snapshot(self, *, max_namespace_bytes: int = 64 * 1024 * 1024) -> tuple[bytes, dict[str, Any]]:
        """今の Agent Step の境界の状態を、1 つの bytes（`capx.snapshot.blob`）にする。

        物理は進めない。生成コードの変数を保存できなければ `NamespaceSnapshotError`、
        Gym が対応していなければ `GymSnapshotError` を投げ、bytes は作らない（片方だけの
        Snapshot を作らない）。元の trial はそのまま続けられる。

        Returns:
            (bytes, 計測値 {"namespace_s", "gym_s", "pack_s"})
        """
        from capx.snapshot import blob, gym_state

        if self._task is None:
            raise RuntimeError("reset_for_trial の前は Snapshot を取れない")
        executor = getattr(self._env, "_process_executor", None)
        if executor is None:
            raise gym_state.GymSnapshotError(
                "Snapshot は生成コードを別プロセスで動かす worker でだけ取れる（ローカル実行は未対応）"
            )

        started = time.monotonic()
        namespace, namespace_info = executor.snapshot(max_bytes=max_namespace_bytes)
        namespace_done = time.monotonic()
        state, gym_info = gym_state.capture(self._env)
        gym_done = time.monotonic()

        local = {
            "trial": self._trial,
            "seed": self._seed,
            "reset_rngs": self._reset_rngs,
            "execution_time_ns": self._execution_time_ns,
            "last_reward": self._last_reward,
            "last_terminated": self._last_terminated,
            "last_truncated": self._last_truncated,
            "last_sandbox_rc": int(self._last_info.get("sandbox_rc", -1)),
            "last_task_completed": _optional_bool(self._last_info.get("task_completed")),
            "steps": [
                {
                    "reward": s.reward,
                    "task_completed": s.task_completed,
                    "terminated": s.terminated,
                    "truncated": s.truncated,
                }
                for s in self._steps
            ],
        }
        history = [
            {
                "code": s.code,
                "ok": s.ok,
                "stdout": s.stdout,
                "stderr": s.stderr,
                "execution_time_s": s.execution_time_s,
            }
            for s in self._steps
        ]
        manifest = {
            "created_at": time.time(),
            "task_id": self._task.task_id,
            "trial": self._trial,
            "seed": self._seed,
            "agent_step": len(self._steps),
            "env_class": gym_info["env_class"],
            "gym": gym_info,
            "namespace": namespace_info,
            "versions": _versions(),
        }
        data = blob.pack(manifest, gym={"env": state, "local_env": local}, namespace=namespace, history=history)
        packed = time.monotonic()
        return data, {
            "namespace_s": round(namespace_done - started, 4),
            "gym_s": round(gym_done - namespace_done, 4),
            "pack_s": round(packed - gym_done, 4),
        }

    def restore(self, data: bytes) -> dict[str, Any]:
        """Snapshot を戻す。新しい trial として reset してから、状態を上書きする。

        reset は Snapshot と同じ trial 番号・seed で行い（TaskSpec とモデルが同じになる）、
        その後 Gym・生成コードの変数・予算の記録を戻す。物理は進めない。途中で失敗したら
        例外を投げる（呼び出し側は、この env で step させてはいけない）。

        Returns:
            Agent に渡してよい再開の情報（TaskSpec・Step 数・それまでの Step）。報酬は入れない。
        """
        from dataclasses import asdict

        from capx.snapshot import blob, gym_state

        snap = blob.unpack(data)
        manifest = snap.manifest
        task_id = self._task_id or type(self._env).__name__
        if manifest.get("task_id") != task_id:
            raise gym_state.GymSnapshotError(
                f"Snapshot のタスク（{manifest.get('task_id')}）と、この worker のタスク（{task_id}）が違う"
            )
        executor = getattr(self._env, "_process_executor", None)
        if executor is None:
            raise gym_state.GymSnapshotError("Snapshot は生成コードを別プロセスで動かす worker でだけ戻せる")

        local = snap.gym["local_env"]
        if local.get("reset_rngs"):
            # 元の trial と同じ乱数で reset する（同じモデルができる）。reset_for_trial が
            # reset の直前に乱数を控え直すので、ここで戻しておけば、その値も元と同じになる。
            self._restore_rngs = local["reset_rngs"]
        task = self.reset_for_trial(int(local["trial"]), local["seed"])
        gym_state.restore(self._env, snap.gym["env"], manifest.get("gym") or {})
        # 生成コードの INPUTS には、復元した時点の観測を渡す。観測を取ると状態が変わる env が
        # ある（nut_assembly は観測のたびに Observable の時計を進める）ので、取ったあとに
        # もう一度戻す。元の trial はこの観測を取っていないので、取らなかった状態に揃える。
        inputs = self._env._get_observation()
        gym_state.restore(self._env, snap.gym["env"], manifest.get("gym") or {})
        names = executor.restore(snap.namespace, inputs, list(self._env._api_functions()))

        self._steps = [
            RecordedStep(
                code=h["code"],
                ok=h["ok"],
                stdout=h["stdout"],
                stderr=h["stderr"],
                execution_time_s=h["execution_time_s"],
                reward=s["reward"],
                task_completed=s["task_completed"],
                terminated=s["terminated"],
                truncated=s["truncated"],
            )
            for h, s in zip(snap.history, local["steps"], strict=True)
        ]
        self._execution_time_ns = int(local["execution_time_ns"])
        self._last_reward = float(local["last_reward"])
        self._last_terminated = bool(local["last_terminated"])
        self._last_truncated = bool(local["last_truncated"])
        self._last_info = {
            "sandbox_rc": int(local["last_sandbox_rc"]),
            "stdout": self._steps[-1].stdout if self._steps else "",
            "stderr": self._steps[-1].stderr if self._steps else "",
            "task_completed": local["last_task_completed"],
        }
        # 録画は復元した時点から録り直す（reset で録った 1 枚目は、上書き前の状態なので捨てる）
        if self._record_video and hasattr(self._env, "enable_video_capture"):
            self._env.enable_video_capture(True, clear=True, wrist_camera=self._wrist_camera)

        return {
            "task": asdict(task),
            "trial": self._trial,
            "seed": self._seed,
            "agent_step": len(self._steps),
            "history": snap.history,
            "namespace_names": names,
            "namespace_excluded": manifest.get("namespace", {}).get("excluded", []),
            "execution_time_used_s": self._execution_time_ns / 1e9,
            "max_steps": self._budget.max_steps,
        }

    def _capture_rngs(self) -> dict[str, Any] | None:
        """reset の直前の乱数を控える。復元で渡された乱数があれば、先にそれに戻す。"""
        try:
            from capx.snapshot import gym_state
        except ImportError:
            return None
        pending, self._restore_rngs = self._restore_rngs, None
        try:
            if pending is not None:
                gym_state.restore_rngs(self._env, pending)
            return gym_state.capture_rngs(self._env)
        except gym_state.GymSnapshotError:
            return None  # Snapshot に対応していない env（実機など）

    def evaluate(self) -> TrialOutcome:
        """採点する。**Bench だけが呼ぶ。** Agent には見せない。

        現行どおり、Agent の実行が終わった時点の状態で採点する。途中で
        一度でも成功した扱いにはしない。
        """
        return TrialOutcome(
            reward=self._last_reward,
            task_completed=_optional_bool(self._last_info.get("task_completed")),
            terminated=self._last_terminated,
            truncated=self._last_truncated,
            sandbox_rc=int(self._last_info.get("sandbox_rc", 1)),
            steps=list(self._steps),
        )

    @property
    def recorded_steps(self) -> list[RecordedStep]:
        """`step()` を通ったものの記録。Agent の自己申告には頼らない。"""
        return list(self._steps)

    @property
    def inner(self) -> CodeExecutionEnvBase:
        """包んでいる env。Bench が動画保存などで触る。Agent には渡さない。"""
        return self._env

    # -- AgentEnv ----------------------------------------------------------

    def step(self, code: str, *, capture_video: bool = False) -> StepResult:
        if self._closed:
            raise RuntimeError("closed env")

        self._check_budget_before(code)

        frame_start = self._frame_count()
        started = time.monotonic_ns()
        _obs, reward, terminated, truncated, info = self._env.step(code)
        elapsed_ns = time.monotonic_ns() - started
        frame_end = self._frame_count()

        self._execution_time_ns += elapsed_ns
        self._last_info = info
        self._last_reward = float(reward)
        self._last_terminated = bool(terminated)
        self._last_truncated = bool(truncated)

        ok = info.get("sandbox_rc", 1) == 0
        step = RecordedStep(
            code=code,
            ok=ok,
            stdout=self._clip(info.get("stdout", "")),
            stderr=self._clip(info.get("stderr", "")),
            reward=float(reward),
            task_completed=_optional_bool(info.get("task_completed")),
            terminated=bool(terminated),
            truncated=bool(truncated),
            execution_time_s=elapsed_ns / 1e9,
            frame_range=(frame_start, frame_end),
        )
        self._steps.append(step)

        used_s = self._execution_time_ns / 1e9
        remaining_s = max(0.0, self._budget.execution_time_s - used_s)
        reason: TruncationReason = "sim_steps" if truncated else "none"
        if remaining_s <= 0.0:
            reason = "execution_budget"

        return StepResult(
            request_id=uuid.uuid4().hex,
            ok=ok,
            stdout=step.stdout,
            stderr=step.stderr,
            truncated=bool(truncated) or remaining_s <= 0.0,
            truncation_reason=reason,
            execution_time_used_s=used_s,
            execution_time_remaining_s=remaining_s,
            sim_steps_used=getattr(self._env.low_level_env, "_sim_step_count", 0),
            video=self._encode_turn_video(frame_start, frame_end) if capture_video else None,
            video_media_type="video/mp4" if capture_video else None,
        )

    def set_frame_listener(self, listener: Any) -> None:
        """録画フレームの通知先を設定する（ストリーミング用。無い env では何もしない）。"""
        low = getattr(self._env, "low_level_env", None)
        if hasattr(low, "set_frame_listener"):
            low.set_frame_listener(listener)

    def _clip(self, text: str) -> str:
        """stdout / stderr を `Budget.max_output_bytes` までに切り詰める。"""
        limit = self._budget.max_output_bytes
        raw = text.encode("utf-8", errors="replace")
        if len(raw) <= limit:
            return text
        head = raw[:limit].decode("utf-8", errors="ignore")
        return f"{head}\n...[出力が上限 {limit} バイトを超えたため切り捨て]"

    def render(self, camera: str = "main") -> bytes:
        """JPEG の bytes を返す。`TaskSpec.cameras` に無い名前は拒否する。"""
        from PIL import Image

        if camera == "wrist":
            frame = self._env.render_wrist()
        else:
            frame = self._env.render()
        if frame is None:
            return b""

        buf = io.BytesIO()
        Image.fromarray(frame).save(buf, format="JPEG", quality=80)
        return buf.getvalue()

    def close(self) -> None:
        self._closed = True

    # -- 内部 --------------------------------------------------------------

    def _check_budget_before(self, code: str) -> None:
        if len(self._steps) >= self._budget.max_steps:
            raise BudgetExceeded(
                "execution_budget",
                f"step の上限 {self._budget.max_steps} に達した",
            )
        if len(code.encode("utf-8")) > self._budget.max_code_bytes:
            raise BudgetExceeded(
                "output_limit",
                f"コードが上限 {self._budget.max_code_bytes} バイトを超えた",
            )
        used_s = self._execution_time_ns / 1e9
        if used_s >= self._budget.execution_time_s:
            raise BudgetExceeded(
                "execution_budget",
                f"実行時間が上限 {self._budget.execution_time_s}s に達した",
            )
        # trial 全体の時間（LLM 待ち・通信待ちを含む）。step を呼ぶ時点で見るので、
        # LLM の応答待ちで止まったままの Agent は止められない。その間 GPU マシン側は
        # 操作が途絶えるので、backend の idle 回収がコンテナを片付ける。
        elapsed_s = time.monotonic() - self._trial_started
        if elapsed_s >= self._budget.trial_wall_clock_s:
            raise BudgetExceeded(
                "wall_clock",
                f"trial 全体の時間が上限 {self._budget.trial_wall_clock_s}s に達した",
            )

    def _frame_count(self) -> int:
        if hasattr(self._env, "get_video_frame_count"):
            return self._env.get_video_frame_count()
        return 0

    def _camera_names(self) -> tuple[str, ...]:
        """low-level env が持つカメラ名。`"robot0_robotview"` を直書きしない。

        simulator ごとに違う（Phase 6 の LIBERO 対応で効く）。
        """
        low = getattr(self._env, "low_level_env", None)
        names = tuple(getattr(low, "render_camera_names", ()) or ())
        if self._wrist_camera:
            names = names + ("wrist",)
        return names or ("main",)

    def _encode_turn_video(self, start: int, end: int) -> bytes | None:
        """そのターン分のフレームを mp4 にする。

        `capture_video=True` のときだけ呼ぶ。VDM が毎ターンの動画を要求する
        経路で使う。
        """
        if end <= start:
            return None
        # 1 枚ずつ戻して書く。バッファは JPEG で持っているので、まとめて戻すと
        # 生の画像が一度に並び、worker のメモリ上限を超えうる。
        if hasattr(self._env, "iter_video_frames_range"):
            frames = self._env.iter_video_frames_range(start, end)
        elif hasattr(self._env, "get_video_frames_range"):
            frames = iter(self._env.get_video_frames_range(start, end))
        else:
            return None

        import tempfile
        from pathlib import Path

        import imageio

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "turn.mp4"
            written = 0
            with imageio.get_writer(path, fps=30) as writer:
                for frame in frames:
                    writer.append_data(frame)
                    written += 1
            if written == 0:
                return None
            return path.read_bytes()


def _versions() -> dict[str, str]:
    """Snapshot に残す版。復元できなかったときの手がかり（一致は求めない）。"""
    import importlib.metadata
    import platform

    out = {"python": platform.python_version()}
    for name in ("mujoco", "robosuite", "libero", "cloudpickle", "numpy"):
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return out


def _optional_bool(value: Any) -> bool | None:
    """シミュレータの真偽値を Python の `bool` にする。分からないとき（None）は None のまま。

    LIBERO の `check_success()` は numpy の真偽値（`numpy.bool_`）を返すことがある
    （numpy 配列の比較を `and` でつないだ値）。そのまま渡すと、ZMQ を通っても numpy の
    型のまま届き、`result.json` に書けない。Bench に渡す前に、ここで揃える。
    """
    return None if value is None else bool(value)
