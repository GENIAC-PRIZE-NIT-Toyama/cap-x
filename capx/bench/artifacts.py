"""Bench が書く成果物。コード・ログ・動画。

`capx/envs/trial.py` から**そのまま移した**もの。挙動は変えていない。
Agent が何を返そうと、`step()` を通ったものを Bench がここで記録する
——自己申告には頼らない。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from capx.utils.video_utils import _write_video

if TYPE_CHECKING:
    from capx.envs.tasks.base import CodeExecutionEnvBase


def _annotate_code_blocks(
    code_blocks: list[str],
    code_block_metadata: list[dict[str, Any]],
) -> str:
    """Join code blocks into a single string with ``# Code block N`` headers."""
    annotated = []
    for i, (block, metadata) in enumerate(zip(code_blocks, code_block_metadata, strict=False)):
        annotated.append(f"# Code block {i}\n{block}")
    return "\n\n".join(annotated)


def _build_log_lines(
    final_code: str,
    info_step: dict[str, Any],
    reward: float,
    terminated: bool,
    truncated: bool,
    num_regenerations: int,
    num_finishes: int,
    num_code_blocks: int,
    *,
    prefix: str = "",
    stderr_override: str | None = None,
) -> list[str]:
    """Build the standard log-line list used for both normal and timeout summaries."""
    stderr = stderr_override if stderr_override is not None else info_step.get("stderr", "")
    lines = ["-" * 100]
    if prefix:
        lines.append(prefix)
    lines.extend([
        "Generated program:",
        final_code if final_code else "(no program available)",
        "\n\nEnvironment response:",
        f"  Sandbox failed: {info_step.get('sandbox_rc', 1)}",
        f"  Stdout: {info_step.get('stdout', '')}",
        f"  Stderr: {stderr}",
        f"  Reward: {reward}",
        f"  Task Completed: {info_step.get('task_completed', False)}",
        f"  Terminated: {terminated}, Truncated: {truncated}",
        f"  Num Regenerations: {num_regenerations}",
        f"  Num Finishes: {num_finishes}",
        f"  Num Code Blocks: {num_code_blocks}",
        "-" * 100,
    ])
    return lines


# ---------------------------------------------------------------------------
# Trial video directory helper
# ---------------------------------------------------------------------------

def _trial_video_dir(
    config: dict[str, Any],
    trial: int,
    info_step: dict[str, Any],
    reward: float,
) -> str:
    """Return the trial output directory path used for video saving."""
    return os.path.join(
        config["output_dir"],
        f"trial_{trial:02d}_sandboxrc_{info_step['sandbox_rc']}_reward_{reward:.3f}"
        f"_taskcompleted_{int(info_step.get('task_completed', False))}",
    )


def _save_trial_video(
    env: CodeExecutionEnvBase,
    config: dict[str, Any],
    trial: int,
    info_step: dict[str, Any],
    reward: float,
    num_code_blocks: int,
    *,
    suffix_extra: str = "",
) -> None:
    """Save recorded video frames from the environment, if available."""
    if not config["record_video"] or not hasattr(env, "get_video_frames"):
        return
    frames = env.get_video_frames(clear=True)
    if not frames or not config["output_dir"]:
        return

    base_dir = _trial_video_dir(config, trial, info_step, reward)
    suffix = f"{reward:.3f}"
    if suffix_extra:
        suffix += f"_{suffix_extra}"

    if isinstance(frames, list):
        _write_video(frames, base_dir, suffix=suffix)
    elif isinstance(frames, dict):
        for key, frame in frames.items():
            _write_video(frame, base_dir, suffix=f"{suffix}_{key}")


def _save_turn_and_combined_videos(
    env: CodeExecutionEnvBase,
    config: dict[str, Any],
    trial: int,
    info_step: dict[str, Any],
    reward: float,
    turn_frame_ranges: list[tuple[int, int]],
) -> None:
    """Save per-turn videos and a combined video of all turns.

    Gets all frames from the environment (clearing the buffer), then writes:
      - ``video_turn_00.mp4``, ``video_turn_01.mp4``, ... for each turn
      - ``video_combined.mp4`` for the full trial
      - If wrist camera is enabled: ``video_turn_00_wrist.mp4``, etc.
    """
    if not config["record_video"] or not config["output_dir"]:
        return
    if not hasattr(env, "get_video_frames"):
        return

    all_frames = env.get_video_frames(clear=True)
    if not all_frames:
        return

    base_dir = _trial_video_dir(config, trial, info_step, reward)

    # all_frames may be a list (Robosuite) or a dict of lists (R1Pro multi-camera).
    # Normalise to a list for slicing; dict case is handled by _write_multi_video.
    if isinstance(all_frames, dict):
        # Multi-camera: write each camera stream as a combined video
        for key, frames in all_frames.items():
            if frames:
                _write_video(frames, base_dir, suffix=f"combined_{key}")
        return

    # Per-turn videos
    for i, (start, end) in enumerate(turn_frame_ranges):
        turn_frames = all_frames[start:end]
        if turn_frames:
            _write_video(turn_frames, base_dir, suffix=f"turn_{i:02d}")

    # Combined video
    _write_video(all_frames, base_dir, suffix="combined")

    # Wrist camera videos
    if config.get("use_wrist_camera") and hasattr(env, "get_wrist_video_frames"):
        wrist_frames = env.get_wrist_video_frames(clear=True)
        if wrist_frames:
            for i, (start, end) in enumerate(turn_frame_ranges):
                wrist_turn = wrist_frames[start:end]
                if wrist_turn:
                    _write_video(wrist_turn, base_dir, suffix=f"turn_{i:02d}_wrist")
            _write_video(wrist_frames, base_dir, suffix="combined_wrist")





# ---------------------------------------------------------------------------
# Remote / 共通: 試行フォルダに、見返すための成果物を書く
# ---------------------------------------------------------------------------


def save_trial_extras(
    trial_dir: str,
    *,
    steps: list,
    llm_entries: list[dict],
    agent_artifacts: dict,
    images: list[bytes],
    result: dict,
) -> None:
    """試行フォルダに次を書く。Local と同じ名前の場所に、Remote でも同じ物が入る。

    - ``result.json`` … 固定スキーマの結果（`capx.bench.schema.TrialResult`）
    - ``steps/`` … 全ステップのコードと stdout / stderr
    - ``prompts_and_responses/`` … LLM の入出力（同梱クライアント経由の分）
    - ``artifacts/`` … Agent が `AgentResult.artifacts` で渡したもの
    - ``videos/`` … ターンごとの動画と、つないだ動画
    - ``images/`` … Agent が `render()` で取った画像
    """
    import json
    from pathlib import Path

    root = Path(trial_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )

    steps_dir = root / "steps"
    steps_dir.mkdir(exist_ok=True)
    for i, step in enumerate(steps, 1):
        (steps_dir / f"step_{i:02d}.py").write_text(step.code, encoding="utf-8")
        log = f"ok: {step.ok}\nexecution_time_s: {step.execution_time_s:.3f}\n"
        log += f"--- stdout ---\n{step.stdout}\n--- stderr ---\n{step.stderr}\n"
        (steps_dir / f"step_{i:02d}.log").write_text(log, encoding="utf-8")

    prompts_dir = root / "prompts_and_responses"
    prompts_dir.mkdir(exist_ok=True)
    for entry in llm_entries:
        n = entry["index"]
        name = "initial_prompt" if n == 0 else f"multi_turn_prompt_{n - 1:02d}"
        (prompts_dir / f"{name}.txt").write_text(
            json.dumps(entry["prompt"], indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (prompts_dir / f"response_{n:02d}.txt").write_text(
            str(entry.get("content") or ""), encoding="utf-8"
        )

    if agent_artifacts:
        art_dir = root / "artifacts"
        art_dir.mkdir(exist_ok=True)
        for name, value in agent_artifacts.items():
            safe = Path(str(name)).name or "artifact"
            data = value if isinstance(value, bytes) else str(value).encode("utf-8")
            (art_dir / safe).write_bytes(data)

    videos = [(i, s.video) for i, s in enumerate(steps, 1) if getattr(s, "video", None)]
    if videos:
        video_dir = root / "videos"
        video_dir.mkdir(exist_ok=True)
        paths = []
        for i, data in videos:
            path = video_dir / f"turn_{i:02d}.mp4"
            path.write_bytes(data)
            paths.append(path)
        _concat_videos(paths, video_dir / "combined.mp4")

    if images:
        image_dir = root / "images"
        image_dir.mkdir(exist_ok=True)
        for i, data in enumerate(images, 1):
            (image_dir / f"render_{i:03d}.jpg").write_bytes(data)


def _json_default(value: Any) -> Any:
    """JSON にできない numpy の値を Python の値にする。

    シミュレータによっては、真偽値や数値を numpy の型で返す（LIBERO の
    `task_completed` がそうだった）。値の出どころ（`LocalAgentEnv`）でも揃えているが、
    今後のシミュレータで漏れても結果を書けるように、ここでも受ける。知らない型は
    これまでどおりエラーにする（文字列にして黙って書くと、おかしな値に気づけない）。
    """
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _concat_videos(paths: list, out) -> None:
    """ターン動画を 1 本につなぐ。同じ設定でエンコードされているので再エンコードしない。

    つなげなくてもターンごとの動画は残るので、失敗しても試行は止めない。
    """
    if len(paths) < 2:
        return
    import subprocess
    import tempfile

    try:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return
    with tempfile.TemporaryDirectory() as tmp:
        listing = f"{tmp}/list.txt"
        with open(listing, "w", encoding="utf-8") as f:
            for p in paths:
                f.write(f"file '{p.resolve()}'\n")
        subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
             "-i", listing, "-c", "copy", str(out)],
            check=False,
        )
