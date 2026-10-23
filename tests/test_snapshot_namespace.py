"""生成コードの Namespace の保存と復元（policy プロセス）。シミュレータは要らない。

元の policy プロセスで定義した変数・関数・クラスを、別の policy プロセスで使い続けられるか。
とくに、関数の globals が復元先の Namespace そのものに結び直されること（後から変数を
書き換えたら関数からも見える）と、API のスタブが新しいプロセスのものに差し替わることを見る。
"""

from __future__ import annotations

import numpy as np
import pytest

from capx.remote_env.worker.executor import ProcessExecutor
from capx.snapshot.namespace import NamespaceSnapshotError

MAX = 64 * 1024 * 1024


def _api():
    calls = []

    def add(a, b):
        calls.append((a, b))
        return a + b

    def pose():
        return np.array([0.1, 0.2, 0.3])

    return {"add": add, "pose": pose}, calls


@pytest.fixture
def pair():
    """元のプロセスと、復元先のプロセス。"""
    a, b = ProcessExecutor(step_timeout_s=30), ProcessExecutor(step_timeout_s=30)
    yield a, b
    a.close()
    b.close()


def _snapshot_and_restore(a, b, fns, code: str, inputs=None):
    a.reset({"k": 1}, list(fns))
    first = a.run(code, {"x": 1}, fns)
    assert first["ok"], first["stderr"]
    data, info = a.snapshot(max_bytes=MAX)
    names = b.restore(data, inputs or {"restored": True}, list(fns))
    return data, info, names


def test_variables_and_functions_survive_a_new_process(pair) -> None:
    a, b = pair
    fns, _ = _api()
    _snapshot_and_restore(a, b, fns, "target = [0.1, 0.2, 0.3]\ndef helper():\n    return target\n")
    assert b.run("print(helper())", {}, fns)["stdout"] == "[0.1, 0.2, 0.3]\n"


def test_function_globals_are_the_live_namespace(pair) -> None:
    """復元した関数は、復元先の Namespace を見る。後から書き換えた値が関数から見える。"""
    a, b = pair
    fns, _ = _api()
    _snapshot_and_restore(a, b, fns, "target = 1\ndef helper():\n    return target\n")
    out = b.run("target = 2\nprint(helper(), helper.__globals__ is globals())", {}, fns)
    assert out["stdout"] == "2 True\n"


def test_numpy_closures_classes_and_shared_references(pair) -> None:
    a, b = pair
    fns, _ = _api()
    code = """
import numpy as np
arr = np.arange(6.0).reshape(2, 3)
def make(k):
    def inner(x):
        return x * k
    return inner
triple = make(3)
class Box:
    def __init__(self, v):
        self.v = v
    def double(self):
        return self.v * 2
box = Box(arr)
same = [box, box]
pts = np.zeros((20000, 3))
"""
    _snapshot_and_restore(a, b, fns, code)
    out = b.run(
        "print(triple(2), box.double()[1, 2], same[0] is same[1], same[0].v is arr,"
        " isinstance(box, Box), pts.shape)",
        {},
        fns,
    )
    assert out["ok"], out["stderr"]
    assert out["stdout"] == "6 10.0 True True True (20000, 3)\n"


def test_user_functions_calling_the_api_use_the_new_stubs(pair) -> None:
    """API（RPC のスタブ）は保存せず、新しいプロセスのスタブに差し替える。"""
    a, b = pair
    fns, calls = _api()
    code = """
def lifted():
    return pose() + add(1, 1)
alias = add
api_list = [add, pose]
"""
    _snapshot_and_restore(a, b, fns, code)
    calls.clear()
    out = b.run("print(lifted().tolist(), alias(2, 3), api_list[0](4, 5), alias is add)", {}, fns)
    assert out["stdout"] == "[2.1, 2.2, 2.3] 5 9 True\n"
    assert calls == [(1, 1), (2, 3), (4, 5)], "呼び出しは新しいプロセスから Gym に届く"


def test_random_state_continues_the_same(pair) -> None:
    """生成コードが乱数を使っても、再開した後の続きは元と同じになる。"""
    a, b = pair
    fns, _ = _api()
    _snapshot_and_restore(a, b, fns, "import random\nimport numpy as np\nrandom.seed(1)\nnp.random.seed(2)\nrandom.random()")
    code = "print(random.random(), np.random.rand())"
    assert a.run(code, {}, fns)["stdout"] == b.run(code, {}, fns)["stdout"]


def test_the_original_process_keeps_working_after_a_snapshot(pair) -> None:
    a, _b = pair
    fns, _ = _api()
    a.reset({}, list(fns))
    a.run("n = 1\ndef f():\n    return n", {}, fns)
    a.snapshot(max_bytes=MAX)
    assert a.run("n += 1\nprint(f(), add(n, 1))", {}, fns)["stdout"] == "2 3\n"


def test_observation_and_inputs_are_not_saved(pair) -> None:
    """`obs` / `INPUTS` は Gym が渡し直す。関数が参照していても中身（画像）を保存しない。"""
    a, b = pair
    fns, _ = _api()
    a.reset({"image": np.zeros((512, 512, 3), dtype=np.uint8)}, list(fns))
    a.run("def peek():\n    return obs['x'], sorted(INPUTS)", {"x": 1, "big": np.zeros(10**6)}, fns)
    data, info = a.snapshot(max_bytes=MAX)
    assert len(data) < 50_000 and "obs" not in info["names"] and "INPUTS" not in info["names"]
    b.restore(data, {"restored": True}, list(fns))
    assert b.run("print(peek())", {"x": 7}, fns)["stdout"] == "(7, ['restored'])\n"


def test_unpicklable_variables_are_reported_not_dropped(pair) -> None:
    a, _b = pair
    fns, _ = _api()
    a.reset({}, list(fns))
    a.run("def g():\n    yield 1\ngen = g()\nimport threading\nlock = threading.Lock()\nok = 1", {}, fns)
    with pytest.raises(NamespaceSnapshotError) as exc:
        a.snapshot(max_bytes=MAX)
    message = str(exc.value)
    assert "gen（generator）" in message and "lock（lock）" in message
    assert "ok（" not in message, "保存できる変数は並べない"
    assert "SNAPSHOT_EXCLUDE" in message
    # 失敗しても、元のプロセスはそのまま使える
    assert a.run("print(ok)", {}, fns)["stdout"] == "1\n"


def test_explicit_exclusion_is_recorded(pair) -> None:
    a, b = pair
    fns, _ = _api()
    a.reset({}, list(fns))
    a.run("def g():\n    yield 1\ngen = g()\nkeep = 5\nSNAPSHOT_EXCLUDE = {'gen'}", {}, fns)
    data, info = a.snapshot(max_bytes=MAX)
    assert info["excluded"] == ["gen"] and "keep" in info["names"]
    b.restore(data, {}, list(fns))
    assert b.run("print(keep, 'gen' in globals())", {}, fns)["stdout"] == "5 False\n"


def test_too_large_names_the_biggest_variables(pair) -> None:
    a, _b = pair
    fns, _ = _api()
    a.reset({}, list(fns))
    a.run("import numpy as np\nbig = np.zeros(400_000)\nsmall = 1", {}, fns)
    with pytest.raises(NamespaceSnapshotError, match="big=3.2MB"):
        a.snapshot(max_bytes=1_000_000)


def test_a_broken_namespace_snapshot_is_rejected(pair) -> None:
    _a, b = pair
    fns, _ = _api()
    with pytest.raises(NamespaceSnapshotError, match="復元できない"):
        b.restore(b"not a pickle", {}, list(fns))


def test_a_dead_process_is_not_snapshotted() -> None:
    fns, _ = _api()
    ex = ProcessExecutor(step_timeout_s=1)
    try:
        ex.reset({}, list(fns))
        ex.run("x = 1", {}, fns)
        ex.run("while True:\n    pass", {}, fns)  # 時間切れでプロセスが止まる（変数は失われる）
        with pytest.raises(NamespaceSnapshotError, match="動いていない"):
            ex.snapshot(max_bytes=MAX)
    finally:
        ex.close()
