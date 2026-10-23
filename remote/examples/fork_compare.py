"""同じ Snapshot から、元のコードと直したコードをそれぞれ別の worker で実行して比べる。

    cd remote
    uv run --env-file .env examples/fork_compare.py --snapshot snap_xxxxxxxx --task libero_object_3 \\
        --original step3_original.py --fixed step3_fixed.py

2 つの Fork は同じ状態から始まり、互いに干渉しない（worker が別）。それぞれ Langfuse に
新しい Trace として残る（tags: fork, fork:human_edit / fork:replay）。結果（reward・成功）を
並べるので、成功・失敗の対（DPO のデータ）や、直したコード（SFT のデータ）を作るのに使える。
Best-of-N にするなら、--fixed を何本も渡す。
"""

from __future__ import annotations

import sys
from pathlib import Path

import tyro

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from forking import RunCode, open_fork, run_fork  # noqa: E402


def main(snapshot: str, task: str, original: str, fixed: list[str], save: bool = True) -> None:
    """
    Args:
        original: 元の Step のコード（ファイル）。同じ状態からやり直すと、元と同じ結果になるはず。
        fixed: 直したコード（ファイル。何本でも）。
    """
    candidates = [("replay", original)] + [("human_edit", path) for path in fixed]
    rows = []
    for fork_type, path in candidates:
        code = Path(path).read_text()
        env = open_fork(snapshot, task)
        try:
            summary = run_fork(env, RunCode([code]), fork_type=fork_type, agent_path=path, save=save)
            rows.append((path, fork_type, summary.task_completed, summary.reward, env.last_snapshot_id))
        finally:
            env.close()

    print("\nコード                     種類         成功    reward   分岐後の Snapshot")
    for path, fork_type, done, reward, snap in rows:
        print(f"{Path(path).name:<26} {fork_type:<12} {str(done):<7} {reward:7.4f}  {snap}")


if __name__ == "__main__":
    tyro.cli(main)
