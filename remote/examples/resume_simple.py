"""Snapshot ID から再開して、コードを 1 つ実行するだけの例。

    cd remote
    uv run --env-file .env examples/resume_simple.py --snapshot snap_xxxxxxxx --task libero_object_3 \\
        --code "print(helper())"

Snapshot ID は Langfuse の env.step の metadata（snapshot_after）か、trial.py の出力で分かる。
再開すると新しい worker が起動し、その Step が終わった時点の Gym と、生成コードの変数・関数が
戻る。元の trial の worker・Trace・Snapshot は変わらない。
"""

from __future__ import annotations

import sys
from pathlib import Path

import tyro

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from forking import open_fork  # noqa: E402


def main(snapshot: str, task: str, code: str = "print(sorted(k for k in globals() if not k.startswith('_')))") -> None:
    env = open_fork(snapshot, task)
    try:
        result = env.step(code)
        print("成功" if result.ok else "エラー")
        print(result.stdout or result.stderr)
        print(f"この Step の Snapshot: {result.snapshot_id}（ここからさらに分岐できる）")
        image = env.render()
        Path("resumed.jpg").write_bytes(image)
        print("今の画像: resumed.jpg")
    finally:
        env.close()


if __name__ == "__main__":
    tyro.cli(main)
