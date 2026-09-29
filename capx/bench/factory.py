"""`AgentEnv` と `Agent` を作る。**Local / Remote の分岐はここだけ。**

`capx/bench/trial.py` は返ってきたものが Local か Remote かを知らないし、
問い合わせる手段もない（`AgentEnv` は `step` / `render` / `close` しか
公開していない）。だから trial のループに `if remote:` が入らない。
"""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
from typing import Any

from capx.agent_api import Agent, AgentEnv


def make_agent_env(
    *,
    env_factory: dict[str, Any] | None = None,
    budget: Any = None,
    record_video: bool = False,
    wrist_camera: bool = False,
    server_url: str | None = None,
    task_id: str | None = None,
) -> AgentEnv:
    """`server_url` があれば Remote、無ければ Local。

    これが唯一の分岐点。Remote は GPU マシンの worker を ZMQ 越しに呼ぶが、
    その worker の中身は同じ `LocalAgentEnv` なので実装は重複しない。
    """
    if server_url:
        from capx.remote_env.client import RemoteAgentEnv

        if not task_id:
            raise ValueError("remote には task_id が要る（env_config は送らない）")
        return RemoteAgentEnv(server_url=server_url, task_id=task_id, budget=budget)

    from capx.envs.configs.instantiate import instantiate
    from capx.local_env import LocalAgentEnv

    if env_factory is None:
        raise ValueError("local には env_factory が要る")
    return LocalAgentEnv(
        instantiate(env_factory),
        budget=budget,
        record_video=record_video,
        wrist_camera=wrist_camera,
    )


def load_agent(spec: str, ctx: Any = None) -> Agent:
    """Agent を読み込む。ファイルパスか `module.Class` を受ける。

    参加者が書くのは 1 ファイル 1 エージェント。規約は「ファイルが `Agent`
    という名前のクラスを定義し、`run(env, task, budget)` を持つ」だけ。
    `__init__(ctx)` は任意で、引数を取らない実装も通す。

    Args:
        spec: `agents/my_agent.py` のようなパス、または
            `capx.baselines.capagent0.CaPAgent0` のような import パス。
        ctx: Agent のコンストラクタに渡す設定。受け取らない Agent もある。
    """
    cls = _resolve(spec)
    return _construct(cls, ctx)


def _resolve(spec: str) -> type:
    path = Path(spec)
    if path.suffix == ".py":
        if not path.exists():
            raise FileNotFoundError(f"Agent ファイルが無い: {path.resolve()}")
        module_name = f"_capx_agent_{path.stem}"
        module_spec = importlib.util.spec_from_file_location(module_name, path)
        if module_spec is None or module_spec.loader is None:
            raise ImportError(f"読み込めない: {path}")
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        return _pick_class(module, str(path))

    module_path, _, attr = spec.rpartition(".")
    if not module_path:
        raise ValueError(f"Agent の指定が不正: {spec!r}")
    module = importlib.import_module(module_path)
    return getattr(module, attr)


def _pick_class(module: Any, where: str) -> type:
    """`Agent` という名前のクラスを探す。無ければ `run` を持つ唯一のクラス。"""
    candidate = getattr(module, "Agent", None)
    if isinstance(candidate, type):
        return candidate

    found = [
        obj
        for name, obj in vars(module).items()
        if isinstance(obj, type)
        and not name.startswith("_")
        and callable(getattr(obj, "run", None))
        and obj.__module__ == module.__name__
    ]
    if len(found) == 1:
        return found[0]
    if not found:
        raise AttributeError(
            f"{where} に Agent クラスが無い。"
            " `class Agent:` を定義し、`run(self, env, task, budget)` を持たせる"
        )
    names = ", ".join(c.__name__ for c in found)
    raise AttributeError(
        f"{where} に run() を持つクラスが複数ある（{names}）。"
        " どれを使うか決められないので、使うものを `Agent` という名前にする"
    )


def _construct(cls: type, ctx: Any) -> Agent:
    """`__init__(ctx)` を取るなら渡し、取らないなら引数なしで作る。

    `__init__` を定義していないクラスは `object.__init__` を継承する。その
    署名は `(self, /, *args, **kwargs)` なので「引数を取る」と読めてしまうが、
    実際に渡すと TypeError になる。継承したものかどうかで判定する。
    """
    if "__init__" not in cls.__dict__:
        return cls()

    try:
        signature = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return cls()

    takes_ctx = any(p.name != "self" for p in signature.parameters.values())
    return cls(ctx) if takes_ctx else cls()
