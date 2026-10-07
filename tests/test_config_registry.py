"""登録済み config が、実在する low_level 環境を指しているかを検査する。

`register_config()` の `low_level` は文字列なので、`register_env()` 側の名前が
変わっても誰も気づかない。ズレていると `get_exec_env(name)(cfg)` が
`_build_low_level()` の中で `KeyError` を投げて落ちる——env の構築時点なので、
oracle も生成コードも一行も走らないまま終わる。

このテストは pytest なしでも動く（`python tests/test_config_registry.py`）。
capx の import が要るので `--extra env` 相当の環境が必要。
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# 2026-09 時点で壊れている登録。直すと挙動が変わる（今まで使えなかったものが
# 使えるようになる）ので、Agent 分離の作業とは切り離して別途対応する。
# 直したらこの集合から消すこと。
KNOWN_BROKEN = {
    "franka_pick_place_code_env",  # low_level="franka_cubes_low_level"
    "franka_pick_place_multi_code_env",  # 同上
    "franka_libero_code_env",  # low_level="franka_libero_low_level"
}


def _scan_sources() -> tuple[set[str], list[tuple[str, str]]]:
    """ソースを読んで (登録済み env 名, [(config 名, その low_level)]) を返す。

    import せずに正規表現で読むのは、simulator の登録が import 時に走り、
    robosuite や LIBERO が入っていない環境では一部しか登録されないため。
    ソースを読めば、どの extra で動かしても同じ結果になる。
    """
    sims = (REPO_ROOT / "capx/envs/simulators/__init__.py").read_text(encoding="utf-8")
    tasks = (REPO_ROOT / "capx/envs/tasks/__init__.py").read_text(encoding="utf-8")

    registered = set(re.findall(r'register_env\("([^"]+)"', sims))

    configs: list[tuple[str, str]] = []
    pattern = re.compile(
        r'register_config\(\s*"([^"]+)",\s*CodeExecEnvConfig\((.*?)\),\s*\)', re.S
    )
    for match in pattern.finditer(tasks):
        name, body = match.group(1), match.group(2)
        low_level = re.search(r'low_level="([^"]+)"', body)
        if low_level:
            configs.append((name, low_level.group(1)))

    return registered, configs


def test_configs_point_at_registered_envs() -> None:
    registered, configs = _scan_sources()

    assert registered, "register_env() が 1 つも見つからない。走査が壊れている"
    assert configs, "register_config() が 1 つも見つからない。走査が壊れている"

    broken = {name: low for name, low in configs if low not in registered}
    unexpected = set(broken) - KNOWN_BROKEN

    assert not unexpected, (
        "config が未登録の low_level を指している: "
        + ", ".join(f"{n} -> {broken[n]!r}" for n in sorted(unexpected))
        + f"。登録済みは {sorted(registered)}"
    )


def test_known_broken_list_is_not_stale() -> None:
    """直したのに KNOWN_BROKEN に残っている、を防ぐ。

    これが落ちたら、直った名前を KNOWN_BROKEN から消す。
    """
    registered, configs = _scan_sources()
    by_name = dict(configs)

    fixed = {
        name
        for name in KNOWN_BROKEN
        if name in by_name and by_name[name] in registered
    }
    assert not fixed, (
        f"KNOWN_BROKEN に残っているが既に直っている: {sorted(fixed)}。"
        " この集合から消すこと"
    )

    missing = {name for name in KNOWN_BROKEN if name not in by_name}
    assert not missing, (
        f"KNOWN_BROKEN の名前が register_config に存在しない: {sorted(missing)}"
    )


if __name__ == "__main__":
    test_configs_point_at_registered_envs()
    test_known_broken_list_is_not_stale()
    print("config registry OK")
