"""同梱の Agent。

参加者が書く `agents/` と 1 文字違いで並ぶのを避けるため、`capx/agents/` では
なくここに置く。

- `CaPAgent0`: 現行 `trial.py` の Agent 部分そのもの。multi-turn・VDM・ensemble。
  比較対象のベースラインになる。
- `OracleAgent`: 人が書いた正解コードを 1 回実行する。環境と API の疎通確認用。
"""

from capx.baselines.capagent0 import CaPAgent0
from capx.baselines.oracle import OracleAgent

__all__ = ["CaPAgent0", "OracleAgent"]
