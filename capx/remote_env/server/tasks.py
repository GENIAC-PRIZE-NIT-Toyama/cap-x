"""参加者が選べるタスクの allowlist。

クライアントから `env_config` を受けない。`_target_` を含む YAML を受理すると
任意の import / instantiate の入口になるので、`task_id` という狭い入力を受けて、
ここで config に展開する。載っていないタスクは作れない。

新しいタスクを出すときは、ここに 1 行足す。
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class TaskEntry:
    config_path: str
    """イメージの中の config（リポジトリ直下からの相対パス）。"""

    runtime: str = "robosuite"
    """どの worker イメージで動かすか。robosuite と libero は同じ venv に
    入らない（`[tool.uv] conflicts`）ので、タスクごとに決める。"""

    overrides: tuple[tuple[str, str], ...] = ()
    """config への上書き（ドット区切りのキー, 値）。worker が読み込み後に当てる。
    LIBERO は 1 本の設定を、suite 名とタスク番号だけ変えて全タスクに使う。
    値はここ（backend）で決める。クライアントからは受けない。"""


TASKS: dict[str, TaskEntry] = {
    "cube_stack": TaskEntry("env_configs/cube_stack/franka_robosuite_cube_stack.yaml"),
    "cube_lifting": TaskEntry("env_configs/cube_lifting/franka_robosuite_cube_lifting.yaml"),
    "cube_restack": TaskEntry("env_configs/cube_restack/franka_robosuite_cube_restack.yaml"),
    "nut_assembly": TaskEntry("env_configs/nut_assembly/franka_robosuite_nut_assembly.yaml"),
    "spill_wipe": TaskEntry("env_configs/spill_wipe/franka_robosuite_spill_wipe.yaml"),
    "two_arm_lift": TaskEntry("env_configs/two_arm_lift/franka_robosuite_two_arm_lift.yaml"),
}

#: LIBERO の標準 5 suite と、それぞれのタスク数。LIBERO-PRO の派生 suite
#: （`*_with_mug` / `*_swap` / `*_ood` など）は含めない。
LIBERO_SUITES: dict[str, int] = {
    "libero_spatial": 10,
    "libero_object": 10,
    "libero_goal": 10,
    "libero_10": 10,
    "libero_90": 90,
}

#: LIBERO の全タスクが使う設定。suite 名と番号だけを上書きする。
LIBERO_CONFIG = "env_configs/libero/franka_libero_spatial_0.yaml"

TASKS.update(
    {
        f"{suite}_{index}": TaskEntry(
            LIBERO_CONFIG,
            runtime="libero",
            overrides=(
                ("env.cfg.low_level.suite_name", suite),
                ("env.cfg.low_level.task_id", str(index)),
            ),
        )
        for suite, count in LIBERO_SUITES.items()
        for index in range(count)
    }
)

#: runtime -> イメージ名。
#: 本番では digest で固定する（タグは付け替えられる）:
#:   CAPX_WORKER_IMAGE=capx-worker@sha256:...
#: digest は `docker image inspect --format '{{.Id}}' capx-worker:latest` で分かる。
IMAGES: dict[str, str] = {
    "robosuite": os.environ.get("CAPX_WORKER_IMAGE", "capx-worker:latest"),
    "libero": os.environ.get("CAPX_WORKER_IMAGE_LIBERO", "capx-worker-libero:latest"),
}


def summarize(names) -> str:
    """タスク名の一覧を、番号つきのものは範囲にまとめて見せる。

    LIBERO だけで 130 あるので、全部を並べると読めない。
    `libero_object_0` … `libero_object_9` は `libero_object_0〜9` にする。
    """
    import re

    groups: dict[str, list[int]] = {}
    plain: list[str] = []
    for name in names:
        match = re.fullmatch(r"(.+)_(\d+)", name)
        if match:
            groups.setdefault(match.group(1), []).append(int(match.group(2)))
        else:
            plain.append(name)
    parts = sorted(plain)
    for prefix in sorted(groups):
        numbers = sorted(groups[prefix])
        if numbers == list(range(numbers[0], numbers[-1] + 1)) and len(numbers) > 1:
            parts.append(f"{prefix}_{numbers[0]}〜{numbers[-1]}")
        else:
            parts.extend(f"{prefix}_{n}" for n in numbers)
    return ", ".join(parts)


def resolve(task_id: str) -> TaskEntry:
    try:
        return TASKS[task_id]
    except KeyError:
        raise KeyError(f"{task_id!r} は選べない。選べるのは: {summarize(TASKS)}") from None
