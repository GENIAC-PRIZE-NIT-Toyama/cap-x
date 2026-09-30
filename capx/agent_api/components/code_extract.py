"""モデルの応答から Python コードを取り出す。

`capx/utils/launch_utils.py` の `_extract_code()` を、自作 Agent からも
使える部品として再公開する。実装はそちらにあり、挙動は変えていない。

Agent はモデルの応答を自分で組み立てるので、抽出も Agent 側の責務になる。
"""

from __future__ import annotations

from capx.utils.launch_utils import _extract_code

__all__ = ["extract_code"]


def extract_code(content: str) -> list[str]:
    """```python フェンスの中身を取り出す。

    フェンスが無ければ応答全体をコードとみなす。空応答は空文字 1 件になる
    （呼び出し側はブロック数が 0 にならない前提で書かれている）。

    Args:
        content: モデルの生の応答。

    Returns:
        コードブロックのリスト。
    """
    return _extract_code(content)
