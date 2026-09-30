"""生成コードを別プロセスで動かす境界。

見たいのは 4 点。API 呼び出しが Gym 側に戻ること、ステップをまたいで変数が残る
こと、無限ループを止められること、worker の秘密が子プロセスに渡らないこと。
"""

from __future__ import annotations

import numpy as np
import pytest

from capx.remote_env.worker import rpc
from capx.remote_env.worker.executor import ProcessExecutor


@pytest.fixture
def executor():
    ex = ProcessExecutor(step_timeout_s=20)
    yield ex
    ex.close()


def _api():
    calls = []

    def add(a, b):
        calls.append((a, b))
        return a + b

    def pose():
        return np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])

    def boom():
        raise ValueError("no object")

    return {"add": add, "pose": pose, "boom": boom}, calls


def test_rpc_keeps_tuples_and_arrays() -> None:
    value = (np.arange(3.0), ("a", 1), [1, 2])
    out = rpc.unpack(rpc.pack(value))
    assert isinstance(out, tuple) and isinstance(out[1], tuple)
    assert isinstance(out[2], list)
    assert np.array_equal(out[0], np.arange(3.0))


def test_api_calls_go_back_to_the_gym_and_variables_persist(executor) -> None:
    fns, calls = _api()
    executor.reset({"k": 1}, list(fns))

    first = executor.run("x = add(1, 2)\nprint(x)\np, q = pose()\nprint(type(pose()).__name__)", {}, fns)
    assert first["ok"] and first["stdout"] == "3\ntuple\n"
    assert calls == [(1, 2)]

    second = executor.run("print(x + 1)", {}, fns)  # 前のステップの変数が残る
    assert second["stdout"] == "4\n"


def test_an_api_error_reaches_the_code_and_the_traceback_is_reported(executor) -> None:
    fns, _ = _api()
    executor.reset({}, list(fns))
    result = executor.run("boom()", {}, fns)
    assert not result["ok"]
    assert "no object" in result["stderr"] and "Traceback" in result["stderr"]


def test_an_infinite_loop_is_stopped_and_the_next_step_still_works() -> None:
    fns, _ = _api()
    ex = ProcessExecutor(step_timeout_s=1)
    try:
        ex.reset({}, list(fns))
        stopped = ex.run("while True:\n    pass", {}, fns)
        assert not stopped["ok"] and "上限" in stopped["stderr"]

        again = ex.run("print('again')", {}, fns)  # 新しいプロセスで動く
        assert again["ok"] and again["stdout"] == "again\n"
    finally:
        ex.close()


def test_worker_secrets_are_not_visible_to_the_generated_code(executor, monkeypatch) -> None:
    monkeypatch.setenv("CAPX_CURVE_SERVER_SECRET", "topsecret")
    fns, _ = _api()
    executor.reset({}, list(fns))
    result = executor.run(
        "import os\nprint(os.environ.get('CAPX_CURVE_SERVER_SECRET'))", {}, fns
    )
    assert result["stdout"] == "None\n"
