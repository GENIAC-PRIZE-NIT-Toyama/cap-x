"""`remote/envfile.py`（`remote/.env` の自動読み込み）。

`import envfile` は読み込みの副作用を持つので、サブプロセスで走らせて見る。
Agent も GPU マシンも使わない。
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REMOTE = Path(__file__).resolve().parent.parent / "remote"

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("dotenv") is None, reason="python-dotenv が無い（remote/ の依存）"
)

# remote/ の下のスクリプトとして、envfile を読み込んだ後の環境変数と argv を出す
SCRIPT = (
    "import json, os, sys\n"
    "import envfile\n"
    "print(json.dumps({'value': os.environ.get('CAPX_ENVFILE_TEST'), 'argv': sys.argv[1:]}))\n"
)


@pytest.fixture
def script():
    # envfile は「remote/ の下から起動したか」を見るので、remote/ の下に置く
    path = REMOTE / f"_envfile_probe_{os.getpid()}.py"
    path.write_text(SCRIPT)
    yield path
    path.unlink()


def _run(script: Path, *args: str, value: str | None = None) -> dict:
    env = {k: v for k, v in os.environ.items() if k != "CAPX_ENVFILE_TEST"}
    if value is not None:
        env["CAPX_ENVFILE_TEST"] = value
    out = subprocess.run(
        [sys.executable, str(script), *args], capture_output=True, text=True, timeout=60, env=env
    )
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


def test_env_file_is_loaded_and_removed_from_argv(script, tmp_path) -> None:
    env_file = tmp_path / "other.env"
    env_file.write_text("CAPX_ENVFILE_TEST=from-file\n")
    assert _run(script, "--env-file", str(env_file), "--task", "x") == {
        "value": "from-file",
        "argv": ["--task", "x"],
    }
    assert _run(script, f"--env-file={env_file}")["value"] == "from-file"


def test_existing_env_wins(script, tmp_path) -> None:
    env_file = tmp_path / "other.env"
    env_file.write_text("CAPX_ENVFILE_TEST=from-file\n")
    assert _run(script, "--env-file", str(env_file), value="from-shell")["value"] == "from-shell"


def test_missing_env_file_stops(script) -> None:
    out = subprocess.run(
        [sys.executable, str(script), "--env-file", "no/such.env"],
        capture_output=True, text=True, timeout=60,
    )
    assert out.returncode != 0
    assert "no/such.env" in out.stderr


def test_import_from_outside_remote_does_not_load(tmp_path) -> None:
    """テストなどが import しただけでは、そのプロセスの環境変数を書き換えない。"""
    env_file = tmp_path / "other.env"
    env_file.write_text("CAPX_ENVFILE_TEST=from-file\n")
    outside = tmp_path / "outside.py"
    outside.write_text(f"import sys\nsys.path.insert(0, {str(REMOTE)!r})\n" + SCRIPT)
    assert _run(outside, "--env-file", str(env_file)) == {
        "value": None,
        "argv": ["--env-file", str(env_file)],
    }
