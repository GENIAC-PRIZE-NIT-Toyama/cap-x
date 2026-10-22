"""いちばん小さい Agent。ここから書き換えて自分の Agent にする。

ルールは 2 つだけ。
  1. `Agent` という名前のクラスを作り、`BaseAgent` を継承する
  2. `run(self, env, task, budget)` を持たせる

`BaseAgent` を継承しておくと、`run` を書き忘れたときに GPU マシンへ繋ぐ前に
エラーで教えてくれる。エディタの補完も効く。

`env.step(code)` でロボットを動かすコードを送ると、GPU マシンで実行されて
結果（stdout / stderr）が返ってくる。`env.render()` で今の画像（JPEG）が取れる。
成功したかどうか（reward）は Agent からは見えない。実行が終わったあと、
`trial.py` が表示する。

LLM を呼んで起きた例外（タイムアウトなど）は捕まえずにそのまま投げる。`suite.py` は
例外の種類で「LLM 側の障害」か「Agent の失敗」かを見分けて、前者ならやり直す。
"""

from capx.agent_api import AgentEnv, AgentResult, BaseAgent, Budget, TaskSpec


class Agent(BaseAgent):
    def run(self, env: AgentEnv, task: TaskSpec, budget: Budget) -> AgentResult:
        # task.instruction … やること / task.api_docs … 使える関数の説明
        goal = next(
            (line for line in task.instruction.splitlines() if line.startswith("Goal")),
            "",
        )
        print("やること:", goal)

        result = env.step("open_gripper()")
        print("実行できた:", result.ok, "  出力:", repr(result.stdout))

        image = env.render()
        print("画像:", len(image), "bytes（JPEG）")

        return AgentResult()
