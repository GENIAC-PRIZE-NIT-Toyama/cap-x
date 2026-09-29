"""CLI 引数を Bench と Agent に分ける。

`LaunchArgs` は Bench 用（trial 数・出力先）と Agent 用（モデル・VDM 設定）を
同じ dataclass に持っていた。誰の設定なのかが型から読めないので分ける。

YAML も同じ形で `bench:` / `agent:` ブロックに分けられる。ただし既存の約 200 本
は最上位に旧キーを置いているので、**そのまま動かす**——`agent:` が無ければ
旧キーから詰め替え、警告を出す。
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass, field
from typing import Any

from capx.agent_api import Budget


@dataclass
class BenchArgs:
    """Agent を差し替えても変わらない部分。"""

    config_path: str
    """環境を定義する YAML のパス。"""

    agent: str = "capx.baselines.capagent0.CaPAgent0"
    """Agent の指定。`agents/my_agent.py` か `module.Class`。"""

    total_trials: int | None = None
    """試行回数。省略時は YAML の `trials`。"""

    num_workers: int | None = None
    """並列ワーカー数。省略時は YAML の `num_workers`。"""

    output_dir: str | None = None
    """成果物の出力先。"""

    record_video: bool | None = None
    """動画を録るか。"""

    seed: int | None = None
    """seed。省略時は trial 番号（trial は 1 始まり）。

    注意: 現状 seed は robosuite まで届いておらず、初期配置は固定されない
    （docs/RemoteDevelopment.md の Phase 0B 節）。
    """

    execution_time_s: float = 1000.0
    """`step()` に使える累計時間。知覚 API 待ちを含み、LLM 待ちを含まない。"""

    trial_wall_clock_s: float = 3000.0
    """trial 全体の上限。LLM 待ち込みの安全網。"""

    max_steps: int = 10
    """`step()` を呼べる回数。現行の `MULTITURN_LIMIT` 相当。"""

    use_oracle_code: bool | None = None
    """oracle を回す。`OracleAgent` に切り替わる。"""

    debug: bool = False

    def budget(self) -> Budget:
        return Budget(
            execution_time_s=self.execution_time_s,
            trial_wall_clock_s=self.trial_wall_clock_s,
            max_steps=self.max_steps,
        )


@dataclass
class AgentArgs:
    """Agent に渡す設定。`--model` などはここ。

    自作 Agent は受け取っても無視してよい（モデル名をファイルに直書きする
    実装も許容する）。
    """

    model: str = ""
    server_url: str | None = None
    api_key: str | None = None
    wire: str | None = None
    temperature: float | None = None
    max_tokens: int = 2048 * 10
    reasoning_effort: str | None = None

    multi_turn_prompt: str | None = None
    use_visual_feedback: bool = False
    use_img_differencing: bool = False
    use_video_differencing: bool = False
    use_wrist_camera: bool = False
    use_parallel_ensemble: bool = False
    use_multimodel: bool = False
    use_legacy_multi_turn_decision_prompt: bool = False
    visual_differencing_model: str | None = None
    visual_differencing_model_server_url: str | None = None
    visual_differencing_model_api_key: str | None = None
    visual_differencing_wire: str | None = None

    extra: dict[str, Any] = field(default_factory=dict)

    def to_context(self, debug: bool = False):
        from capx.baselines.capagent0 import AgentContext

        data = asdict(self)
        data.pop("extra", None)
        return AgentContext(debug=debug, extra=dict(self.extra), **data)


# 最上位に置かれていた旧 Agent キー。`agent:` ブロックが無いときに詰め替える。
_LEGACY_AGENT_KEYS = (
    "multi_turn_prompt",
    "use_visual_feedback",
    "use_img_differencing",
    "use_video_differencing",
    "use_wrist_camera",
    "use_parallel_ensemble",
    "use_multimodel",
    "use_legacy_multi_turn_decision_prompt",
    "visual_differencing_model",
    "visual_differencing_model_server_url",
    "visual_differencing_model_api_key",
    "visual_differencing_wire",
    "server_url",
    "wire",
)


def agent_args_from_config(
    configs_dict: dict[str, Any],
    overrides: AgentArgs | None = None,
) -> AgentArgs:
    """YAML から `AgentArgs` を組む。`agent:` が無ければ旧キーから詰め替える。

    既存の約 200 本の YAML を書き換えずに動かすための互換レイヤ。
    """
    block = configs_dict.get("agent")

    if block is None:
        legacy = {
            key: configs_dict[key]
            for key in _LEGACY_AGENT_KEYS
            if key in configs_dict
        }
        if legacy:
            warnings.warn(
                "YAML の最上位に Agent 設定があります"
                f"（{', '.join(sorted(legacy))}）。"
                " `agent:` ブロックへ移してください。当面は読み続けます。",
                DeprecationWarning,
                stacklevel=2,
            )
        block = legacy

    # multi_turn_prompt は env.cfg に置かれていることもある（設計資料の指摘）
    if "multi_turn_prompt" not in block:
        env_cfg = (configs_dict.get("env") or {}).get("cfg") or {}
        if isinstance(env_cfg, dict) and env_cfg.get("multi_turn_prompt"):
            warnings.warn(
                "`multi_turn_prompt` が env.cfg にあります。"
                " Agent のプロトコルなので `agent:` ブロックへ移してください。",
                DeprecationWarning,
                stacklevel=2,
            )
            block = {**block, "multi_turn_prompt": env_cfg["multi_turn_prompt"]}

    known = {f for f in AgentArgs.__dataclass_fields__ if f != "extra"}
    kwargs = {k: v for k, v in block.items() if k in known}
    extra = {k: v for k, v in block.items() if k not in known}

    args = AgentArgs(**kwargs, extra=extra)

    if overrides is not None:
        for name in known:
            value = getattr(overrides, name)
            default = AgentArgs.__dataclass_fields__[name].default
            if value != default:  # CLI が明示したものだけ上書き
                setattr(args, name, value)

    return args
