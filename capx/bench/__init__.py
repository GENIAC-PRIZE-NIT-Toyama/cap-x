"""Bench: seed・予算・成果物・指標・評価。

Agent を差し替えても変わらない部分がここに集まる。Local でも Remote でも
同じこの実装を使うので、参加者のスコアと研究側の数字が同じ定義になる。

`capx.envs`（Gym）を module 冒頭では import しない——手元PC の軽量
インストールから読めるようにするため。Gym が要るものは関数の中で遅延 import する。
"""

from capx.bench.factory import load_agent, make_agent_env
from capx.bench.trial import run_trial

__all__ = ["load_agent", "make_agent_env", "run_trial"]
