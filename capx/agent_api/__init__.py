"""Agent と Bench が共有する契約。

**この package は Gym（`capx.envs`）を import しない。** `capx.envs.__init__` は
import 時に simulator registration を走らせるので、ここから触れると手元PC 向けの
軽量インストール（`remote/`）が成立しなくなる。

`tests/test_import_hygiene.py` がこれを検査している。
"""

from capx.agent_api.types import (
    PROTOCOL_VERSION,
    Agent,
    AgentEnv,
    AgentResult,
    Budget,
    BudgetExceeded,
    EnvUnavailable,
    FailureKind,
    StepResult,
    TaskSpec,
    TruncationReason,
)

__all__ = [
    "PROTOCOL_VERSION",
    "Agent",
    "AgentEnv",
    "AgentResult",
    "Budget",
    "BudgetExceeded",
    "EnvUnavailable",
    "FailureKind",
    "StepResult",
    "TaskSpec",
    "TruncationReason",
]
