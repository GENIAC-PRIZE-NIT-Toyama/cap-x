"""`capx.agent_api` が重い依存を引かないことを検査する。

これが崩れると手元PC 向けの軽量インストール（`remote/`）が成立しなくなる。
`capx/envs/__init__.py` は import 時に simulator registration を走らせるので、
`capx.agent_api` から `capx.envs` に触れた瞬間に robosuite / torch まで芋づるで入る。

このテストは **capx がまったくインストールされていない環境でも意味を持つ**ので、
`tests/` の中でも特に軽い。CI では `cd remote && uv run pytest` からも動かせる。
"""

from __future__ import annotations

import subprocess
import sys

# `capx.agent_api` を import した直後の `sys.modules` に現れてはいけないもの。
# simulator・学習系・重い数値計算のいずれか 1 つでも入っていたら、
# 依存の向きがどこかで逆転している。
FORBIDDEN = {
    "torch",
    "torchvision",
    "open3d",
    "ray",
    "transformers",
    "robosuite",
    "mujoco",
    "libero",
    "sam3",
    "pyroki",
    "trimesh",
    "viser",
    "cv2",
    "capx.envs",
}


def _import_in_subprocess(module: str) -> set[str]:
    """別プロセスで import し、読み込まれた `sys.modules` の名前を返す。

    同一プロセスだと、他のテストが先に重い module を読んでいた場合に
    偽陰性になる（既に `sys.modules` にあるので「引いていない」と誤判定する）。
    """
    code = (
        "import sys, json;"
        f"import {module};"
        "print(json.dumps(sorted(sys.modules)))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    import json

    return set(json.loads(out.stdout))


def test_agent_api_is_light() -> None:
    loaded = _import_in_subprocess("capx.agent_api")
    leaked = FORBIDDEN & loaded
    assert not leaked, (
        f"capx.agent_api が重い依存を引いている: {sorted(leaked)}。"
        " agent_api は Gym を import してはいけない（docs/RemoteDevelopment.md §11）。"
    )


def test_agent_api_has_no_numpy() -> None:
    """型だけの module なので numpy すら要らない。

    画像は PNG/JPEG の bytes で受け渡すので、`capx.agent_api.types` の時点では
    numpy に触れない。ここが崩れると `remote/` の依存が 1 段重くなる。
    """
    loaded = _import_in_subprocess("capx.agent_api.types")
    assert "numpy" not in loaded, (
        "capx.agent_api.types が numpy を引いている。"
        " 画像は bytes で受け渡す契約（docs/RemoteDevelopment.md §2）。"
    )


def test_components_are_light() -> None:
    """自作 Agent が使う部品も simulator を引かない。

    `capx/utils/launch_utils.py` が module 冒頭で `capx.envs` を import して
    いたため、コード抽出を 1 つ使うだけで robosuite まで芋づるで入っていた。
    Gym が要る関数の中で遅延 import するよう変えた。ここが戻ると
    `remote/` に simulator が入る。
    """
    loaded = _import_in_subprocess("capx.agent_api.components.code_extract")
    leaked = FORBIDDEN & loaded
    assert not leaked, (
        f"code_extract が重い依存を引いている: {sorted(leaked)}"
    )


def test_step_result_has_no_reward_field() -> None:
    """`StepResult` に採点結果が混ざっていないことを型レベルで固定する。

    レビューだけに頼ると、あとから「デバッグ用に」足されて気づかれない。
    報酬を Agent に見せないという決定（docs/RemoteDevelopment.md §5）を
    テストで守る。
    """
    from dataclasses import fields

    from capx.agent_api import StepResult

    names = {f.name for f in fields(StepResult)}
    forbidden = {"reward", "task_completed", "info", "obs", "observation", "env"}
    leaked = names & forbidden
    assert not leaked, f"StepResult に採点結果が混ざっている: {sorted(leaked)}"


def test_agent_env_protocol_surface() -> None:
    """`AgentEnv` が `step` / `render` / `close` だけを公開していることを固定する。

    `reset` や `evaluate` が生えると、Agent が自分で採点できてしまう。
    """
    from capx.agent_api import AgentEnv

    public = {
        name
        for name in vars(AgentEnv)
        if not name.startswith("_")
    }
    assert public == {"step", "render", "close"}, (
        f"AgentEnv の公開面が変わっている: {sorted(public)}。"
        " reset / evaluate は Bench の責務（docs/RemoteDevelopment.md §2）。"
    )
