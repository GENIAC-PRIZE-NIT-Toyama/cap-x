"""生成コードの Namespace（policy プロセスの globals）を bytes にし、別の policy プロセスで戻す。

**policy プロセスの中だけで使う。** 復元は pickle を解くので任意のコードが動く。
policy プロセスはもともと生成コード（任意のコード）を動かす場所なので、ここで解く限り
新しい危険は増えない。Gym のプロセスや backend では解かない。

cloudpickle で保存する。`exec` で作った関数やクラスは `__module__ == "__main__"` なので、
cloudpickle は参照ではなく中身（コード）ごと保存する。

**関数の globals を、復元先の生きた Namespace に結び直す。** cloudpickle は関数ごとに、
参照している globals だけを集めた新しい dict を作って復元する。そのままだと、復元した
`helper()` は Namespace とは別の dict を見るので、次の Step で `target = ...` と書き換えても
`helper()` は古い値を返す。保存時に「globals が Namespace そのものである関数」を見つけ、
Namespace を persistent id（`NAMESPACE_ID`）にして、復元時に生きた Namespace に差し替える。

**API の関数（RPC のスタブ）は保存しない。** スタブは前の policy プロセスのソケットを
掴んでいる。スタブは persistent id（`("capx:api", 名前)`）にして、復元時に新しいプロセスの
スタブに差し替える。ユーザーの関数が API を呼んでいても、リストに入れていても同じ。
"""

from __future__ import annotations

import io
import pickle
import types
from collections.abc import Callable
from typing import Any

#: 生きた Namespace を指す persistent id。
NAMESPACE_ID = "capx:namespace"
#: API のスタブを指す persistent id の頭。`(API_ID, "goto_pose")`。
API_ID = "capx:api"
#: Gym が渡し直す値（`obs` / `INPUTS`）を指す persistent id の頭。関数が参照していても
#: 中身（画像を含む観測）を保存せず、復元先の値に差し替える。
SYSTEM_ID = "capx:system"
_SYSTEM_VALUES = ("obs", "INPUTS")
#: スタブに付ける印（関数の属性）。値は API 名。
API_ATTR = "__capx_api__"
#: Namespace にこの名前で名前の集まりを置くと、それらを保存しない（明示的な除外）。
EXCLUDE_NAME = "SNAPSHOT_EXCLUDE"
#: 保存しない名前。Gym が実行のたびに渡し直すものと、Python が足すもの。
#: `INPUTS` は reset 時の観測（画像を含み大きい）。復元後は、復元した時点の観測が入る。
SYSTEM_NAMES = frozenset({"__builtins__", "__name__", "obs", "INPUTS"})

PICKLE_PROTOCOL = 4


class NamespaceSnapshotError(RuntimeError):
    """Namespace を保存・復元できない。メッセージに、どの変数がなぜだめかを書く。"""


def is_stub(value: Any, name: str | None = None) -> bool:
    """`value` が API のスタブか。`name` を渡すと、その名前のスタブかどうか。"""
    if type(value) is not types.FunctionType:
        return False
    api = value.__dict__.get(API_ATTR)
    return api is not None and (name is None or api == name)


def _pickler_class():
    import cloudpickle

    class _Pickler(cloudpickle.Pickler):
        def __init__(self, file: Any, namespace: dict[str, Any]) -> None:
            super().__init__(file, protocol=PICKLE_PROTOCOL)
            self._namespace = namespace
            # id で引くので、dict / list のような実体だけを対象にする（小さな int などは
            # 別の変数と同じ id を持ちうる）
            self._system = {
                id(namespace[key]): key
                for key in _SYSTEM_VALUES
                if isinstance(namespace.get(key), (dict, list))
            }

        def persistent_id(self, obj: Any) -> Any:
            if obj is self._namespace:
                return NAMESPACE_ID
            if self._system and id(obj) in self._system:
                return (SYSTEM_ID, self._system[id(obj)])
            if type(obj) is types.FunctionType:
                api = obj.__dict__.get(API_ATTR)
                if api is not None:
                    return (API_ID, api)
            return None

        # cloudpickle の内部（2.1 と 3.x で同じ形）。関数を作り直すときの globals を、
        # Namespace そのものにする（上の persistent id で生きた Namespace に戻る）。
        def _function_getnewargs(self, func: Any) -> Any:
            args = super()._function_getnewargs(func)
            if func.__globals__ is self._namespace:
                return (args[0], self._namespace, *args[2:])
            return args

    return _Pickler


def _dumps(value: Any, namespace: dict[str, Any]) -> bytes:
    buf = io.BytesIO()
    _pickler_class()(buf, namespace).dump(value)
    return buf.getvalue()


def selected_names(namespace: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """保存する変数と、明示的に除外した名前。"""
    excluded = _explicit_exclusions(namespace)
    names = {
        key: value
        for key, value in namespace.items()
        if key not in SYSTEM_NAMES and key not in excluded and not is_stub(value, key)
    }
    return names, sorted(excluded)


def _explicit_exclusions(namespace: dict[str, Any]) -> set[str]:
    raw = namespace.get(EXCLUDE_NAME)
    if raw is None:
        return set()
    if isinstance(raw, str) or not all(isinstance(x, str) for x in _iter(raw)):
        raise NamespaceSnapshotError(
            f"{EXCLUDE_NAME} は名前（文字列）の集まりにする。例: {EXCLUDE_NAME} = {{'gen', 'f'}}"
        )
    return set(raw)


def _iter(raw: Any) -> list[Any]:
    try:
        return list(raw)
    except TypeError:
        return [raw]


def dump_namespace(namespace: dict[str, Any], *, max_bytes: int) -> tuple[bytes, dict[str, Any]]:
    """Namespace を bytes にする。

    保存できない変数があれば、**黙って外さずに** `NamespaceSnapshotError` を投げる。
    メッセージには変数ごとの理由を書く。

    Returns:
        (bytes, {"names": 保存した名前, "excluded": SNAPSHOT_EXCLUDE で外した名前})
    """
    names, excluded = selected_names(namespace)
    try:
        raw = _dumps({"names": names, "random": _random_state()}, namespace)
    except Exception as exc:
        raise NamespaceSnapshotError(_diagnose(names, namespace, exc)) from None
    if len(raw) > max_bytes:
        raise NamespaceSnapshotError(_too_large(names, namespace, len(raw), max_bytes))
    return raw, {"names": sorted(names), "excluded": excluded}


def _diagnose(names: dict[str, Any], namespace: dict[str, Any], exc: Exception) -> str:
    """1 つずつ保存してみて、だめな変数と理由を並べる。"""
    failures = []
    for key, value in names.items():
        try:
            _dumps({key: value}, namespace)
        except Exception as one:
            failures.append(f"  - {key}（{type(value).__name__}）: {type(one).__name__}: {one}")
    head = "生成コードの変数を Snapshot に保存できない。"
    if not failures:
        return f"{head}{type(exc).__name__}: {exc}"
    return (
        head
        + "保存できない変数（関数の場合は、その関数が参照している変数が原因のこともある）:\n"
        + "\n".join(failures[:20])
        + f"\n要らなければ `del 変数名` で消すか、`{EXCLUDE_NAME} = {{'変数名'}}` で明示的に除外する。"
        "ソケット・ファイル・スレッド・ジェネレータなどは保存できない"
    )


def _too_large(names: dict[str, Any], namespace: dict[str, Any], size: int, limit: int) -> str:
    sizes = []
    for key, value in names.items():
        try:
            sizes.append((len(_dumps({key: value}, namespace)), key))
        except Exception:
            continue
    top = ", ".join(f"{key}={n / 1e6:.1f}MB" for n, key in sorted(sizes, reverse=True)[:5])
    return (
        f"生成コードの変数が大きすぎて Snapshot に保存できない（{size / 1e6:.1f}MB > 上限 "
        f"{limit / 1e6:.1f}MB）。大きい順: {top}。要らなければ `del` するか `{EXCLUDE_NAME}` で除外する"
    )


def load_namespace(
    raw: bytes, namespace: dict[str, Any], make_stub: Callable[[str], Any]
) -> list[str]:
    """`dump_namespace` の bytes を解き、`namespace` に書き戻す。戻した名前を返す。

    `make_stub(name)` は新しい policy プロセスでの API のスタブを返す。
    """

    class _Unpickler(pickle.Unpickler):
        def persistent_load(self, pid: Any) -> Any:
            if pid == NAMESPACE_ID:
                return namespace
            if isinstance(pid, tuple) and len(pid) == 2 and pid[0] == API_ID:
                return make_stub(str(pid[1]))
            if isinstance(pid, tuple) and len(pid) == 2 and pid[0] == SYSTEM_ID:
                return namespace.get(str(pid[1]))
            raise pickle.UnpicklingError(f"知らない persistent id: {pid!r}")

    try:
        data = _Unpickler(io.BytesIO(raw)).load()
    except Exception as exc:
        raise NamespaceSnapshotError(
            f"生成コードの変数を復元できない: {type(exc).__name__}: {exc}"
        ) from None
    if not isinstance(data, dict) or not isinstance(data.get("names"), dict):
        raise NamespaceSnapshotError("Namespace の Snapshot の形が違う")
    namespace.update(data["names"])
    _set_random_state(data.get("random") or {})
    return sorted(data["names"])


def _random_state() -> dict[str, Any]:
    """policy プロセスの乱数（`random` と、読み込まれていれば `numpy.random`）。

    生成コードが乱数を使っても、再開した後に同じ続きになるようにする。
    """
    import random
    import sys

    state: dict[str, Any] = {"random": random.getstate()}
    numpy = sys.modules.get("numpy")
    if numpy is not None:
        state["numpy"] = numpy.random.get_state()
    return state


def _set_random_state(state: dict[str, Any]) -> None:
    import random

    if "random" in state:
        random.setstate(state["random"])
    if "numpy" in state:
        import numpy

        numpy.random.set_state(state["numpy"])
