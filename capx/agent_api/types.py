"""Agent が触れてよいものだけを定義する型。

この module は **Agent と Bench の両方が import する唯一の契約**であり、
Gym（`capx.envs`）を import しない。`capx.envs.__init__` は import 時に
simulator registration を走らせるため、ここから触れると手元PC 向けの軽量
インストール（`remote/`）が成立しなくなる。

依存は標準ライブラリのみ。numpy すら import しない（画像は PNG/JPEG の
bytes で受け渡す）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, Protocol, runtime_checkable

# 契約のバージョン。クライアントと worker が別々に更新されうるので、
# セッション確立時に突き合わせる。
PROTOCOL_VERSION = 1


class FailureKind(str, Enum):
    """失敗の種類。リトライ方針がこれで決まるので、文字列判定をしない。

    - ``INFRASTRUCTURE``: コンテナ異常終了・ZMQ 切断・GPU OOM・セッション作成失敗。
      Agent と無関係なのでリトライする（新しいセッションを作り直す）。
    - ``BUDGET``: ``execution_time_s`` / ``trial_wall_clock_s`` の超過。
      Agent の責任なのでリトライしない。
    - ``AGENT_ERROR``: Agent 自身が送出した例外（LLM 認証失敗など）。
    """

    INFRASTRUCTURE = "infrastructure"
    BUDGET = "budget"
    AGENT_ERROR = "agent_error"


TruncationReason = Literal[
    "none",
    "execution_budget",  # execution_time_s 超過
    "wall_clock",  # trial_wall_clock_s 超過
    "sim_steps",  # low-level env の max_steps 到達
    "output_limit",  # stdout/stderr がサイズ上限に達した
    "infrastructure",  # 通信断など
]


@dataclass(frozen=True)
class TaskSpec:
    """Agent に渡すタスク記述。Gym の実体も報酬も含まない。

    Attributes:
        task_id: ``"cube_stack"`` のようなタスク識別子。
        seed: 初期配置を決める seed。trial 番号がそのまま入る（trial は 1 始まり）。
        instruction: 解決済みのタスク文。LIBERO の goal のように実行時にしか
            決まらないものも、env 側で解決してからここに入る。
        api_docs: 生成コードから呼べる関数の docstring を連結したもの
            （``ApiBase.combined_doc()``）。
        default_prompt: 現行 ``obs["full_prompt"]`` 相当。ベースライン用で、
            Agent は無視して自分で組み立ててよい。
        cameras: ``render()`` に渡せるカメラ名。worker が埋めるので、
            クライアントは ``"robot0_robotview"`` のような値を直書きしない。
        reference_code: oracle コード。Bench が明示的に許可したときだけ入る。
    """

    task_id: str
    seed: int
    instruction: str
    api_docs: str
    default_prompt: list[dict[str, Any]] = field(default_factory=list)
    cameras: tuple[str, ...] = ("main",)
    reference_code: str | None = None
    protocol_version: int = PROTOCOL_VERSION


@dataclass(frozen=True)
class Budget:
    """Agent に渡す予算。案内であると同時に、worker/backend が権威的に強制する。

    Agent が budget を無視する前提で設計すること。

    Attributes:
        execution_time_s: ``step(code)`` の処理に使える累計時間。知覚 API 待ちを
            含み、**LLM 待ちを含まない**。比較用の上限で、既定は現行の
            ``TRIAL_TIMEOUT_SECONDS`` と同じ 1000 秒。
        trial_wall_clock_s: trial 全体の上限。LLM 待ち・通信待ち込みの安全網。
        max_steps: ``step()`` を呼べる回数。現行の ``MULTITURN_LIMIT`` 相当。
        max_code_bytes: 1 回の ``step()`` に渡せるコードのサイズ上限。
        max_output_bytes: 1 回の ``step()`` が返す stdout / stderr それぞれの上限。
            超えた分は切り捨てる（生成コードが大量に print しても、メモリと通信を
            埋められないようにする）。
    """

    execution_time_s: float = 1000.0
    trial_wall_clock_s: float = 3000.0
    max_steps: int = 10
    max_code_bytes: int = 64 * 1024
    max_output_bytes: int = 64 * 1024


@dataclass
class StepResult:
    """``AgentEnv.step()`` の返り値。

    **ここに入れてはいけないもの**: ``reward`` / ``task_completed`` /
    Gym の生 ``info`` / low-level observation / 内部 config /
    セッション管理用 credential。採点は Bench だけが読む。

    Attributes:
        ok: コードが例外なく走ったか（現行の ``sandbox_rc == 0``）。
        truncated: 何らかの上限に達して打ち切られたか。
        truncation_reason: 打ち切りの理由。``truncated`` が False なら ``"none"``。
        execution_time_used_s: この trial でこれまでに使った実行時間の累計。
        execution_time_remaining_s: 残り。Agent はこれを見て戦略を変えられる。
        video: ``capture_video=True`` のときだけ入る。mp4 の bytes。
        video_media_type: ``"video/mp4"`` など。形式を推測させない。
    """

    request_id: str
    ok: bool
    stdout: str
    stderr: str
    truncated: bool = False
    truncation_reason: TruncationReason = "none"
    execution_time_used_s: float = 0.0
    execution_time_remaining_s: float = 0.0
    sim_steps_used: int = 0
    video: bytes | None = None
    video_media_type: str | None = None


@dataclass
class AgentResult:
    """Agent が ``run()`` の最後に返すもの。すべて任意。

    採点には使わない。Bench は ``step()`` に渡された全コードを自動で記録するので、
    ``artifacts`` は追加分だけでよい。

    token 数は同梱の LLM クライアントを使った場合のみ Bench が自動集計する。
    自前で SDK を呼ぶ Agent はここで自己申告するが、**参考値**であり強制できない。
    """

    notes: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str | bytes] = field(default_factory=dict)
    llm_calls: int | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None


class BudgetExceeded(RuntimeError):
    """予算を使い切ったときに ``AgentEnv`` が送出する。

    Bench はこれを ``FailureKind.BUDGET`` として扱い、**リトライしない**。
    その時点の状態で採点する。
    """

    def __init__(self, reason: TruncationReason, message: str = "") -> None:
        super().__init__(message or f"budget exceeded: {reason}")
        self.reason: TruncationReason = reason


class EnvUnavailable(RuntimeError):
    """コンテナ異常終了・通信断など、Agent と無関係な失敗。

    Bench はこれを ``FailureKind.INFRASTRUCTURE`` として扱い、
    **新しいセッションを作り直して**リトライする。中断後の env は
    一貫性の保証がないため再利用しない。
    """


class CapacityFull(EnvUnavailable):
    """GPU マシンの同時セッションが上限。しばらく待てば空く。"""


class EnvStartFailed(EnvUnavailable):
    """セッションを作れなかった（選べないタスク・コンテナの起動失敗など）。"""


@runtime_checkable
class AgentEnv(Protocol):
    """Agent から見た環境。``step`` と ``render`` しか公開しない。

    実体は 2 つ。``LocalAgentEnv`` は同一プロセスの ``CodeExecutionEnvBase`` を
    包み、``RemoteAgentEnv`` は ZMQ 越しに GPU マシンの worker を呼ぶ。
    **Agent からは区別できない**し、区別する手段も与えない。

    ``reset()`` と ``evaluate()`` は意図的に含まれていない。どちらも Bench の
    責務であり、Agent に報酬を見せないための境界がここにある。
    """

    def step(self, code: str, *, capture_video: bool = False) -> StepResult:
        """Python コードを実行する。

        Raises:
            BudgetExceeded: 予算を使い切った。
            EnvUnavailable: 環境側の障害。
        """
        ...

    def render(self, camera: str = "main") -> bytes:
        """現在の画像を JPEG の bytes で返す。

        ``camera`` は ``TaskSpec.cameras`` に含まれる名前。
        """
        ...

    def close(self) -> None:
        """セッションを閉じる。Bench が呼ぶ。"""
        ...


@runtime_checkable
class Agent(Protocol):
    """参加者が書くもの。1 ファイル 1 エージェント。

    規約はこれだけ——ファイルが ``Agent`` という名前のクラスを定義し、
    ``run(env, task, budget)`` を持つこと。``__init__(ctx)`` は任意で、
    受け取らない実装も許容する。

    Example:
        >>> class Agent:
        ...     def __init__(self, ctx):
        ...         self.ctx = ctx
        ...
        ...     def run(self, env, task, budget):
        ...         code = my_llm(task.default_prompt, self.ctx)
        ...         env.step(code)
        ...         return AgentResult()
    """

    def run(self, env: AgentEnv, task: TaskSpec, budget: Budget) -> AgentResult:
        ...
