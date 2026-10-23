"""実機（GPU マシンの backend・Docker の worker・MinIO）で、Snapshot の受け入れシナリオを通す。

    cd remote
    uv run --env-file .env ../scripts/snapshot_e2e_check.py --task cube_stack_low
    uv run --env-file .env ../scripts/snapshot_e2e_check.py --task libero_object_0_low

Low-Level 版のタスク（関節角を直接動かす関数がある）で、知覚サーバに頼らずに確かめる。

1. Snapshot を有効にして回す。Step 1 でヘルパー関数を定義し、Step 2 で使い、Step 3 で失敗する
2. 元の worker を終わらせる
3. Step 2 の Snapshot から新しい worker で再開し、Step 1 の関数で別のコードを実行して続ける
4. 同じ Snapshot から別の worker で元の Step 3 をやり直し、元と同じ結果になるか比べる
5. Fork の Snapshot からさらに分岐する
6. 元の Snapshot が変わっていないか（記録の sha256）、不正な ID が断られるか
"""

from __future__ import annotations

import os
import sys
import time

import requests

STEP1 = """\
import numpy as np
home = np.asarray(get_observation()["robot_joint_pos"][:7], dtype=np.float64)
target = home + np.array([0.15, 0.1, 0.0, 0.1, 0.0, 0.1, 0.0])
def helper(scale):
    return home + (target - home) * scale
class Memo:
    def __init__(self):
        self.log = []
memo = Memo()
memo.log.append("step1")
print(repr(np.round(home, 6).tolist()))
"""
STEP2 = """\
move_to_joints(helper(0.5))
close_gripper()
memo.log.append("step2")
print(repr(get_observation()["robot_joint_pos"][:7].tolist()))
"""
STEP3_ORIGINAL = """\
move_to_joints(helper(2.0))
print(repr(get_observation()["robot_joint_pos"][:7].tolist()))
raise RuntimeError("missed the object")
"""
STEP3_FIXED = """\
move_to_joints(helper(1.0))
open_gripper()
memo.log.append("fixed3")
print(memo.log)
"""
STEP4 = """\
memo.log.append("step4")
print(memo.log)
"""


def main(task: str = "cube_stack_low") -> None:
    from capx.remote_env.client import RemoteAgentEnv, SnapshotError, _auth_headers

    url = os.environ["CAPX_ENV_SERVER_URL"].rstrip("/")
    results: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        results.append((name, ok, detail))
        print(f"{'OK' if ok else 'NG'} {name} {detail}", flush=True)

    def env(**kw):
        t = time.monotonic()
        e = RemoteAgentEnv(server_url=url, task_id=task, timeout_s=1500, **kw)
        return e, time.monotonic() - t

    def record(sid: str) -> dict:
        r = requests.get(f"{url}/snapshots/{sid}", headers=_auth_headers(), timeout=60)
        r.raise_for_status()
        return r.json()

    # 1. 元の trial
    original, created = env(snapshot_enabled=True)
    print(f"セッション作成 {created:.1f}s")
    original.reset_for_trial(1)
    check("reset 直後の Snapshot", bool(original.initial_snapshot_id), original.initial_snapshot_id or original.last_snapshot_error or "")
    steps = []
    for code in (STEP1, STEP2, STEP3_ORIGINAL):
        r = original.step(code)
        steps.append(r)
        print(f"  step{len(steps)} ok={r.ok} snapshot={r.snapshot_id} {r.stdout.strip()[:120]}")
    check("Step 1〜3 の結果", [r.ok for r in steps] == [True, True, False], str([r.ok for r in steps]))
    check("全 Step の Snapshot（例外の Step も）", all(r.snapshot_id for r in steps), original.last_snapshot_error or "")
    outcome = original.evaluate()
    step2 = steps[1].snapshot_id
    sha_before = record(step2)["sha256"]
    rec = record(step2)
    print(f"  Snapshot の大きさ {rec['size_bytes'] / 1024:.1f}KB、保存の時間 {[t['save_s'] for t in original.snapshot_timings]}")

    # 2. 元の worker を終わらせる
    original.close()

    # 3. Step 2 から再開して、直したコードで続ける
    fork, resume_s = env(resume_from=step2, snapshot_enabled=True, record_video=True)
    info = fork.resume_info or {}
    check(
        "再開（新しい worker）",
        fork.agent_step == 2 and "helper" in info.get("namespace_names", []),
        f"{resume_s:.1f}s（うち復元 {info.get('restore_s')}s）",
    )
    fixed = fork.step(STEP3_FIXED)
    check("Step 1 の関数・変数で続ける", fixed.ok and fixed.stdout.startswith("['step1', 'step2', 'fixed3']"), fixed.stdout.strip() or fixed.stderr[-300:])
    check("再開後の動画・画像", bool(fixed.video) and bool(fork.render()))
    more = fork.step(STEP4)
    check("Multi-turn で続ける", more.stdout.strip() == "['step1', 'step2', 'fixed3', 'step4']", more.stdout.strip())
    fork_snapshot = fixed.snapshot_id
    fork.close()

    # 4. 同じ Snapshot から元の Step 3 をやり直す
    replay, _ = env(resume_from=step2)
    again = replay.step(STEP3_ORIGINAL)
    same = again.stdout == steps[2].stdout
    check("元の Step 3 をやり直すと同じ結果（関節角）", same, "" if same else f"\n    元   {steps[2].stdout.strip()}\n    再現 {again.stdout.strip()}")
    replayed = replay.evaluate()
    check(
        "reward・成功判定も同じ",
        abs(replayed.reward - outcome.reward) < 1e-9 and replayed.task_completed == outcome.task_completed,
        f"{outcome.reward} / {replayed.reward}",
    )
    replay.close()

    # 5. Fork の Snapshot からさらに分岐
    deeper, _ = env(resume_from=fork_snapshot)
    out = deeper.step(STEP4).stdout.strip()
    check("Fork からさらに分岐", out == "['step1', 'step2', 'fixed3', 'step4']", out)
    deeper.close()

    # 6. 元の Snapshot は変わらない・不正な ID は断る
    check("元の Snapshot は変わらない", record(step2)["sha256"] == sha_before)
    try:
        env(resume_from="snap_" + "0" * 24)
        check("無い Snapshot は断る", False)
    except SnapshotError as exc:
        check("無い Snapshot は断る", "404" in str(exc), str(exc)[:120])

    failed = [name for name, ok, _ in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} OK" + (f"  NG: {failed}" if failed else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
