"""Low-Level Primitives 版のタスク（`<タスク名>_low`）の配線。シミュレータは要らない。

見たいのは 3 点。High 版と同じ config・runtime を使い、`env.cfg.apis` だけが違うこと。
その上書きが実際の config に当たること。指す API が登録済みであること。
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from capx.remote_env.server import tasks
from capx.remote_env.worker.server import apply_overrides

REPO = Path(__file__).resolve().parent.parent


def test_every_high_level_task_but_two_arm_lift_has_a_low_level_twin() -> None:
    high = {name for name in tasks.TASKS if not name.endswith(tasks.LOW_SUFFIX)}
    low = {name for name in tasks.TASKS if name.endswith(tasks.LOW_SUFFIX)}
    assert low == {f"{name}_low" for name in high - {"two_arm_lift"}}
    assert len(low) == 5 + 130


@pytest.mark.parametrize(
    ("name", "api"),
    [
        ("cube_stack", "FrankaControlApiReduced"),
        ("nut_assembly", "FrankaControlNutAssemblyApiReduced"),
        ("spill_wipe", "FrankaControlSpillWipeApiReduced"),
        ("libero_90_46", "FrankaLiberoApiReduced"),
    ],
)
def test_low_level_task_only_swaps_the_apis(name, api) -> None:
    high, low = tasks.TASKS[name], tasks.TASKS[f"{name}_low"]
    assert (low.config_path, low.runtime) == (high.config_path, high.runtime)
    assert low.overrides == (*high.overrides, ("env.cfg.apis", f"[{api}]"))


@pytest.mark.parametrize("name", ["cube_lifting_low", "nut_assembly_low", "libero_10_3_low"])
def test_overrides_replace_the_apis_of_the_real_config(name) -> None:
    entry = tasks.TASKS[name]
    config = yaml.safe_load((REPO / entry.config_path).read_text(encoding="utf-8"))
    apply_overrides(config, [f"{key}={value}" for key, value in entry.overrides])
    assert config["env"]["cfg"]["apis"] == [tasks.LOW_LEVEL_APIS[name.removesuffix("_low")]]


def test_low_level_apis_are_registered() -> None:
    """`capx.integrations` を import すると重いので、登録している行を読む。"""
    source = (REPO / "capx/integrations/__init__.py").read_text(encoding="utf-8")
    registered = set(re.findall(r'register_api\(\s*"(\w+)"', source))
    assert set(tasks.LOW_LEVEL_APIS.values()) <= registered


def test_unknown_low_level_task_is_rejected() -> None:
    with pytest.raises(KeyError, match="two_arm_lift_low"):
        tasks.resolve("two_arm_lift_low")
