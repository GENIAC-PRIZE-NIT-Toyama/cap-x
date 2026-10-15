"""LIBERO タスクを remote で選べるようにする配線。シミュレータは要らない。

見たいのは 4 点。標準 5 suite の全タスクが許可リストにあること、LIBERO 用の image で
起動されること、suite 名と番号が backend の値として worker に渡ること、worker がその値を
config に当てること。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import yaml
import zmq

from capx.remote_env.server import docker as dk
from capx.remote_env.server import tasks
from capx.remote_env.server.sessions import Config, SessionManager
from capx.remote_env.worker.server import apply_overrides

REPO = Path(__file__).resolve().parent.parent


def test_all_tasks_of_the_five_standard_suites_are_selectable() -> None:
    libero = {k: v for k, v in tasks.TASKS.items() if v.runtime == "libero"}
    assert len(libero) == 130
    for suite, count in tasks.LIBERO_SUITES.items():
        assert f"{suite}_0" in libero and f"{suite}_{count - 1}" in libero
        assert f"{suite}_{count}" not in libero
    assert "libero_object_with_mug_0" not in tasks.TASKS, "派生 suite は含めない"
    assert tasks.TASKS["cube_stack"].runtime == "robosuite", "robosuite のタスクはそのまま"


def test_each_libero_task_names_its_suite_and_index() -> None:
    entry = tasks.TASKS["libero_object_3"]
    assert dict(entry.overrides) == {
        "env.cfg.low_level.suite_name": "libero_object",
        "env.cfg.low_level.task_id": "3",
    }
    assert (REPO / entry.config_path).exists()


def test_libero_runs_on_its_own_image(monkeypatch) -> None:
    import importlib

    assert tasks.IMAGES["libero"] == "capx-worker-libero:latest"
    monkeypatch.setenv("CAPX_WORKER_IMAGE_LIBERO", "capx-worker-libero@sha256:abc")
    try:
        assert importlib.reload(tasks).IMAGES["libero"] == "capx-worker-libero@sha256:abc"
    finally:
        monkeypatch.delenv("CAPX_WORKER_IMAGE_LIBERO")
        importlib.reload(tasks)


class _FakeDocker:
    def __init__(self) -> None:
        self.commands: list[list[str]] = []

    async def __call__(self, cmd):
        self.commands.append(cmd)
        if cmd[:3] == ["docker", "network", "inspect"]:
            return 1, "", ""
        return 0, "", ""


def test_a_libero_session_starts_the_libero_image_with_its_task() -> None:
    docker = _FakeDocker()
    manager = SessionManager(Config(port_start=19500, port_end=19502), runner=docker)
    key = zmq.curve_keypair()[0].decode("ascii")
    session = asyncio.run(manager.create("alice", "libero_goal_7", key))

    run = next(c for c in docker.commands if c[:2] == ["docker", "run"])
    assert "capx-worker-libero:latest" in run
    after_image = run[run.index("capx-worker-libero:latest") + 1 :]
    assert after_image[0] == tasks.LIBERO_CONFIG
    assert after_image[after_image.index("--task-id") + 1] == "libero_goal_7"
    overrides = after_image[after_image.index("--override") + 1 :]
    assert "env.cfg.low_level.suite_name=libero_goal" in overrides
    assert "env.cfg.low_level.task_id=7" in overrides
    assert session.task_id == "libero_goal_7"


def test_robosuite_sessions_get_no_overrides() -> None:
    spec = dk.RunSpec(
        session_id="s", owner="o", image="capx-worker:latest",
        config_path="env_configs/cube_stack/franka_robosuite_cube_stack.yaml",
        host_port=19500, task_id="cube_stack",
    )
    cmd = dk.run_command(spec)
    assert "--override" not in cmd
    assert cmd[cmd.index("--task-id") + 1] == "cube_stack"


def test_overrides_are_applied_to_the_real_libero_config() -> None:
    config = yaml.safe_load((REPO / tasks.LIBERO_CONFIG).read_text(encoding="utf-8"))
    apply_overrides(
        config,
        ["env.cfg.low_level.suite_name=libero_90", "env.cfg.low_level.task_id=42"],
    )
    low = config["env"]["cfg"]["low_level"]
    assert low["suite_name"] == "libero_90"
    assert low["task_id"] == 42, "番号は数値として入る"


def test_a_mistyped_override_is_an_error_not_a_new_key() -> None:
    config = {"env": {"cfg": {"low_level": {"task_id": 0}}}}
    with pytest.raises(KeyError):
        apply_overrides(config, ["env.cfg.low_levl.task_id=3"])
    with pytest.raises(KeyError):
        apply_overrides(config, ["env.cfg.low_level.taskid=3"])
    with pytest.raises(ValueError):
        apply_overrides(config, ["env.cfg.low_level.task_id"])


def test_libero_image_differs_from_the_robosuite_one_only_where_it_must() -> None:
    robosuite = (REPO / "docker/worker/Dockerfile").read_text(encoding="utf-8")
    libero = (REPO / "docker/worker/Dockerfile.libero").read_text(encoding="utf-8")
    assert "uv sync --frozen --no-dev --extra libero" in libero
    assert "--extra robosuite" not in libero
    # インストール後のパッケージ名は `libero`（`libero.libero` は無い）
    assert 'python -c "import libero"' in libero and "LIBERO_CONFIG_PATH=/app/.libero" in libero
    assert "USER 10001:10001" in libero
    strip = lambda text: text.split("RUN python - <<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    assert strip(libero) == strip(robosuite), "oracle の除去は同じ処理"

    ignore = (REPO / "docker/worker/Dockerfile.libero.dockerignore").read_text(encoding="utf-8")
    lines = {ln.strip() for ln in ignore.splitlines()}
    assert "capx/third_party/robosuite" in lines
    assert "capx/third_party/LIBERO-PRO" not in lines
    assert "capx/third_party/libero_dependencies" not in lines
    assert {"capx/baselines", "env_configs/human_oracle_code"} <= lines


def test_the_goal_placeholder_is_filled_on_reset() -> None:
    """`{libero_environment_goal}` を実際のゴール文に置き換える（要 gymnasium）。"""
    pytest.importorskip("gymnasium")
    from types import SimpleNamespace

    from capx.envs.tasks.base import CodeExecutionEnvBase

    env = CodeExecutionEnvBase.__new__(CodeExecutionEnvBase)
    env.low_level_env = SimpleNamespace(handle=SimpleNamespace(task_language="put the bowl on the plate"))
    env._task_prompt = "Goal: {libero_environment_goal}\nUse {executed_code} later"
    env._full_prompt = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": [{"type": "text", "text": "Goal: {libero_environment_goal}"}]},
    ]
    env._fill_task_goal()
    assert env._task_prompt.startswith("Goal: put the bowl on the plate")
    assert "{executed_code}" in env._task_prompt, "他の置き場所は触らない"
    assert env._full_prompt[1]["content"][0]["text"] == "Goal: put the bowl on the plate"


def test_task_lists_are_summarized_by_range() -> None:
    text = tasks.summarize(tasks.TASKS)
    assert "libero_90_0〜89" in text and "libero_object_0〜9" in text
    assert "cube_stack" in text
    assert "libero_object_3" not in text, "範囲にまとめる"
    assert tasks.summarize(["a_1", "a_3"]) == "a_1, a_3", "飛び番は並べる"


def test_both_images_let_the_non_root_worker_read_the_urdf_cache() -> None:
    """キャッシュは root が clone し、worker は 10001 で動く。git に読ませる設定が要る。"""
    for name in ("Dockerfile", "Dockerfile.libero"):
        text = (REPO / "docker/worker" / name).read_text(encoding="utf-8")
        assert (
            "git config --system --add safe.directory "
            "/app/.cache/robot_descriptions/example-robot-data"
        ) in text, name
