"""Snapshot の大きさと速さを測る。

    # GPU マシンの backend で（remote/.env の CAPX_ENV_SERVER_URL を使う）
    cd remote && uv run --env-file .env ../scripts/snapshot_bench.py --task cube_stack
    cd remote && uv run --env-file .env ../scripts/snapshot_bench.py --task libero_object_0

    # 手元で（backend と worker を手元のプロセスで動かす。robosuite か LIBERO が要る）
    python scripts/snapshot_bench.py --local --task snap_robosuite

測るもの:
- Step の時間（Snapshot OFF / ON）
- 1 つの Snapshot の大きさ（Gym / 生成コードの変数 / 履歴、圧縮前と後）
- 保存の時間（worker で作る・backend が取り出す・保存先に置く・クライアントが待つ全体）
- 再開（Fork）の時間（新しい worker の起動 + 復元）と、そのうちの復元だけの時間
- 生成コードの変数に点群・画像を持ったときの増え方

結果は表で出し、`--out` に JSON でも書く。
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent

STEP_DEFINE = """\
import numpy as np
target = [0.1, 0.2, 0.3]
def helper(scale=1.0):
    return [x * scale for x in target]
close_gripper()
"""
STEP_MOVE = ["open_gripper()", "close_gripper()"]
STEP_POINTS = "pts = np.random.rand(12000, 3)\nopen_gripper()"
STEP_IMAGE = "img = np.random.randint(0, 255, (512, 800, 3), dtype=np.uint8)\nclose_gripper()"


def _record(server_url: str, snapshot_id: str) -> dict[str, Any]:
    import requests

    from capx.remote_env.client import _auth_headers

    response = requests.get(f"{server_url}/snapshots/{snapshot_id}", headers=_auth_headers(), timeout=60)
    response.raise_for_status()
    return response.json()


def _env(server_url: str, task: str, **kw):
    from capx.remote_env.client import RemoteAgentEnv

    return RemoteAgentEnv(server_url=server_url, task_id=task, timeout_s=1200, **kw)


def bench(server_url: str, task: str, steps: int, forks: int) -> dict[str, Any]:
    out: dict[str, Any] = {"task": task, "steps": steps}

    # -- Step の時間: OFF と ON ------------------------------------------------
    for enabled in (False, True):
        started = time.monotonic()
        env = _env(server_url, task, snapshot_enabled=enabled)
        created = time.monotonic() - started
        env.reset_for_trial(1)
        times = []
        ids = []
        for i in range(steps):
            code = STEP_DEFINE if i == 0 else STEP_MOVE[i % 2]
            t = time.monotonic()
            result = env.step(code)
            times.append(time.monotonic() - t)
            ids.append(result.snapshot_id)
        key = "on" if enabled else "off"
        out[f"session_create_s_{key}"] = round(created, 3)
        out[f"step_s_{key}"] = [round(x, 3) for x in times]
        if enabled:
            out["snapshot_timings"] = env.snapshot_timings
            out["snapshot_ids"] = ids
            # 点群・画像を変数に持ったとき
            grown = {}
            for name, code in (("points_12000x3_float64", STEP_POINTS), ("image_512x800x3_uint8", STEP_IMAGE)):
                result = env.step(code)
                grown[name] = _record(server_url, result.snapshot_id)["sections"]
            out["namespace_growth"] = grown
        env.close()

    record = _record(server_url, out["snapshot_ids"][-1])
    out["sections"] = record["sections"]
    out["snapshot_bytes"] = record["size_bytes"]

    # -- 再開（Fork）の時間 -----------------------------------------------------
    resumes = []
    for _ in range(forks):
        t = time.monotonic()
        fork = _env(server_url, task, resume_from=out["snapshot_ids"][-1])
        elapsed = time.monotonic() - t
        first = fork.step(STEP_MOVE[0])
        resumes.append(
            {
                "fork_create_s": round(elapsed, 3),
                "restore_s": (fork.resume_info or {}).get("restore_s"),
                "first_step_ok": first.ok,
            }
        )
        fork.close()
    out["resume"] = resumes
    return out


def _kb(n: int | None) -> str:
    return "-" if n is None else f"{n / 1024:.1f}KB"


def report(result: dict[str, Any]) -> str:
    timings = result["snapshot_timings"]
    worker = [t["worker_s"] for t in timings if t.get("worker_s") is not None]
    save = [t["save_s"] for t in timings]
    fetch = [t["backend_timing"]["fetch_s"] for t in timings if t.get("backend_timing")]
    put = [t["backend_timing"]["put_s"] for t in timings if t.get("backend_timing")]
    sections = result["sections"]
    lines = [
        f"## {result['task']}",
        "",
        "| 項目 | 値 |",
        "| --- | --- |",
        f"| Gym（圧縮前 → 後） | {_kb(sections['gym']['raw_bytes'])} → {_kb(sections['gym']['compressed_bytes'])} |",
        f"| 生成コードの変数（圧縮前 → 後） | {_kb(sections['namespace']['raw_bytes'])} → {_kb(sections['namespace']['compressed_bytes'])} |",
        f"| 履歴（圧縮前 → 後） | {_kb(sections['history']['raw_bytes'])} → {_kb(sections['history']['compressed_bytes'])} |",
        f"| Snapshot 全体 | {_kb(result['snapshot_bytes'])} |",
        f"| worker で作る（中央値） | {statistics.median(worker) * 1000:.0f}ms |",
        f"| backend が取り出す（中央値） | {statistics.median(fetch) * 1000:.0f}ms |",
        f"| 保存先に置く（中央値） | {statistics.median(put) * 1000:.0f}ms |",
        f"| クライアントが待つ保存全体（中央値） | {statistics.median(save) * 1000:.0f}ms |",
        f"| Step（OFF、中央値） | {statistics.median(result['step_s_off'][1:]):.2f}s |",
        f"| Step（ON、中央値。保存を含む） | {statistics.median(result['step_s_on'][1:]):.2f}s |",
        f"| セッション作成（通常） | {result['session_create_s_off']:.1f}s |",
        f"| 再開（Fork 作成、中央値） | {statistics.median(r['fork_create_s'] for r in result['resume']):.1f}s |",
        f"| うち復元だけ（中央値） | {statistics.median(r['restore_s'] for r in result['resume']):.2f}s |",
    ]
    for name, sec in result["namespace_growth"].items():
        lines.append(
            f"| 変数に {name} を足す | 変数 {_kb(sec['namespace']['raw_bytes'])} → 圧縮後 {_kb(sec['namespace']['compressed_bytes'])} |"
        )
    return "\n".join(lines)


def main(task: str, local: bool = False, steps: int = 6, forks: int = 2, out: str | None = None) -> None:
    """
    Args:
        local: backend と worker を手元で動かす（tests/snapshot_harness.py）。
        steps: 測る Step の数（1 つ目は変数の定義）。
        forks: 再開を何回測るか。
    """
    backend = None
    if local:
        import tempfile

        sys.path.insert(0, str(ROOT / "tests"))
        from snapshot_harness import Backend

        base = Path(tempfile.mkdtemp(prefix="snapshot_bench_"))
        (base / "logs").mkdir()
        backend = Backend(base / "store", base / "logs").start()
        server_url = backend.url
    else:
        server_url = os.environ.get("CAPX_ENV_SERVER_URL", "")
        if not server_url:
            sys.exit("CAPX_ENV_SERVER_URL が無い（--local で手元で測るか、.env を読ませる）")
    try:
        result = bench(server_url.rstrip("/"), task, steps, forks)
    finally:
        if backend is not None:
            backend.stop()
    print(report(result))
    if out:
        Path(out).write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"\nJSON: {out}")


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
