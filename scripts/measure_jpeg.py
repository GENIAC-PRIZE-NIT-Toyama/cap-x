"""JPEG の品質ごとのサイズとエンコード時間を測る（Phase 4A-7）。

GPU マシンで、robosuite の代表タスクを実際に動かしてフレームを集める:

    uv run --no-sync --active scripts/measure_jpeg.py

出力は markdown の表。`docs/RemoteDevelopment.md` §6 に貼る。
"""

from __future__ import annotations

import io
import statistics
import time

import numpy as np
import tyro
from PIL import Image


def _collect(config_path: str, n_frames: int) -> list[np.ndarray]:
    from capx.envs.configs.instantiate import instantiate
    from capx.envs.configs.loader import DictLoader
    from capx.local_env import LocalAgentEnv

    cfg = DictLoader.load([config_path])
    env = LocalAgentEnv(instantiate(cfg["env"]), record_video=True)
    frames: list[np.ndarray] = []
    trial = 0
    while len(frames) < n_frames:
        trial += 1
        env.reset_for_trial(trial)
        # 動きのある場面を撮る。oracle のコードで一通り動かす。
        code = getattr(env.inner, "oracle_code", None) or "open_gripper()\nclose_gripper()"
        env.step(code)
        frames.extend(env.inner.get_video_frames(clear=True))
        if trial > 20:
            break
    return frames[:n_frames]


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * q))]


def main(
    config_path: str = "env_configs/cube_stack/franka_robosuite_cube_stack.yaml",
    /,
    *,
    frames: int = 1000,
) -> None:
    images = _collect(config_path, frames)
    print(f"{len(images)} 枚、{images[0].shape[1]}x{images[0].shape[0]}\n")
    print("| 形式 | p50 (KB) | p95 (KB) | max (KB) | encode p50 (ms) | encode p95 (ms) |")
    print("|---|---|---|---|---|---|")

    def row(label: str, fmt: str, **kw) -> None:
        sizes, times = [], []
        for frame in images:
            image = Image.fromarray(np.ascontiguousarray(frame))
            buf = io.BytesIO()
            start = time.perf_counter()
            image.save(buf, format=fmt, **kw)
            times.append((time.perf_counter() - start) * 1000)
            sizes.append(len(buf.getvalue()) / 1024)
        print(
            f"| {label} | {statistics.median(sizes):.1f} | {_percentile(sizes, 0.95):.1f} | "
            f"{max(sizes):.1f} | {statistics.median(times):.1f} | {_percentile(times, 0.95):.1f} |"
        )

    for quality in (70, 80, 90):
        row(f"JPEG q{quality}", "JPEG", quality=quality)
    row("PNG", "PNG")


if __name__ == "__main__":
    tyro.cli(main)
