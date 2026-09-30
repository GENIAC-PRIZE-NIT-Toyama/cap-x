"""参加者が選べるタスクの allowlist。

クライアントから `env_config` を受けない。`_target_` を含む YAML を受理すると
任意の import / instantiate の入口になるので、`task_id` という狭い入力を受けて、
ここで config に展開する。載っていないタスクは作れない。

新しいタスクを出すときは、ここに 1 行足す。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskEntry:
    config_path: str
    """イメージの中の config（リポジトリ直下からの相対パス）。"""

    runtime: str = "robosuite"
    """どの worker イメージで動かすか。robosuite と libero は同じ venv に
    入らない（`[tool.uv] conflicts`）ので、タスクごとに決める。"""


TASKS: dict[str, TaskEntry] = {
    "cube_stack": TaskEntry("env_configs/cube_stack/franka_robosuite_cube_stack.yaml"),
    "cube_lifting": TaskEntry("env_configs/cube_lifting/franka_robosuite_cube_lifting.yaml"),
    "cube_restack": TaskEntry("env_configs/cube_restack/franka_robosuite_cube_restack.yaml"),
    "nut_assembly": TaskEntry("env_configs/nut_assembly/franka_robosuite_nut_assembly.yaml"),
    "spill_wipe": TaskEntry("env_configs/spill_wipe/franka_robosuite_spill_wipe.yaml"),
    "two_arm_lift": TaskEntry("env_configs/two_arm_lift/franka_robosuite_two_arm_lift.yaml"),
}

#: runtime -> イメージ名。LIBERO は Phase 6 で足す。
IMAGES: dict[str, str] = {
    "robosuite": "capx-worker:latest",
}


def resolve(task_id: str) -> TaskEntry:
    try:
        return TASKS[task_id]
    except KeyError:
        raise KeyError(
            f"{task_id!r} は選べない。選べるのは: {', '.join(sorted(TASKS))}"
        ) from None
