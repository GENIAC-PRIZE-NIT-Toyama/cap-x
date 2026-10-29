"""`remote/.env` を読み込む。各スクリプトの先頭で `import envfile` するだけでよい。

    uv run trial.py                          # remote/.env を読む（無ければ何もしない）
    uv run trial.py --env-file other.env     # .env 以外を使うときだけ指定する

- すでに設定されている環境変数は上書きしない（`OPENAI_BASE_MODEL=xxx uv run ...` や
  `uv run --env-file .env ...` もそのまま使える）。
- `--env-file` は tyro より先にここで取り出して `sys.argv` から消す。関数の既定値
  （`model: str = os.environ.get(...)`）は import 時に決まるので、読むのも import 時にする。
- 読むのは `remote/` の下のスクリプトとして起動したときだけ。テストなどが `suite` を
  import しただけでは読まない（そのプロセスの環境変数を書き換えてしまうため）。
- python-dotenv は remote/ の依存。ルートの環境（tests/ がサブプロセスで走らせるとき）には
  無いことがあるので、そのときは `--env-file` を明示したときだけ止める。
- Windows では、読む前に UTF-8 モードで起動し直す（`ensure_utf8()`）。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # ルートの環境で走らせたとき
    load_dotenv = None

HERE = Path(__file__).resolve().parent
DEFAULT = HERE / ".env"


def _pop_env_file(argv: list[str]) -> str | None:
    for i, arg in enumerate(argv[1:], start=1):
        if arg == "--env-file":
            if i + 1 >= len(argv):
                sys.exit("--env-file の後にファイルを指定する")
            path = argv[i + 1]
            del argv[i : i + 2]
            return path
        if arg.startswith("--env-file="):
            del argv[i]
            return arg.split("=", 1)[1]
    return None


def _run_as_remote_script(argv: list[str]) -> bool:
    return bool(argv and argv[0]) and HERE in Path(argv[0]).resolve().parents


def load(argv: list[str] = sys.argv) -> Path | None:
    """`--env-file` か `remote/.env` を読み、読んだファイルを返す。"""
    if not _run_as_remote_script(argv):
        return None
    given = _pop_env_file(argv)
    if given is not None:
        path = Path(given).expanduser()
        if not path.exists():
            sys.exit(f"--env-file のファイルが無い: {given}")
    elif DEFAULT.is_file():
        path = DEFAULT
    else:
        return None
    if load_dotenv is None:
        if given is None:
            return None
        sys.exit("--env-file を使うには python-dotenv が要る（`cd remote && uv sync`）")
    load_dotenv(path, override=False)
    return path


def ensure_utf8(argv: list[str] = sys.argv) -> None:
    """Windows では Python の UTF-8 モードで起動し直す。

    Windows の既定の文字コードは cp932 で、`capx` が結果を保存する `write_text()` は
    encoding を指定しないため、LLM の出力に cp932 に無い文字（`ž` など）があると
    保存で `UnicodeEncodeError` になる。UTF-8 モードは起動後には切り替えられないので、
    `PYTHONUTF8=1` を付けて同じコマンドを子プロセスで走らせ、その終了コードで終わる。
    """
    if os.name != "nt" or sys.flags.utf8_mode or not _run_as_remote_script(argv):
        return
    env = {**os.environ, "PYTHONUTF8": "1"}
    proc = subprocess.Popen([sys.executable, "-X", "utf8", *argv], env=env)
    while True:
        try:
            sys.exit(proc.wait())
        except KeyboardInterrupt:  # Ctrl+C は子にも届くので、子が終わるのを待つ
            continue


ensure_utf8()
load()
