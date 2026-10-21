"""評価スイートの集計。`suite.py` が最後に呼ぶ。単体でも使える。

    cd remote
    uv run suite_summary.py outputs/<モデル名>/suite_<名前>/<日時>

実行フォルダにある試行ごとの `result.json` と `infra_events.jsonl` だけから計算する。
メモリ上の結果は使わないので、途中で止まった実行も、過去の実行も同じやり方で
集計し直せる。

指標:
- ``pass_at_1`` … Zero-shot pass@1。`task_completed` が true の試行 / 有効な試行
- ``error_rate`` … 推論の失敗に遭遇した試行 / 有効な試行
- ``wall_clock_s`` … 試行全体の時間（LLM 待ちを含む）の平均・中央値・p90
- ``infra.failure_rate`` … インフラの失敗で有効にならなかった試行の割合（参考）

有効な試行 = `result.json` がある試行。インフラの失敗でやり直しを使い切った試行は
分母に入れない。
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

import tyro
import yaml

SCHEMA_VERSION = 1

# 失敗の分類。`failure.kind`（`capx.bench.errors.explain` の分類）で決める。
#: 推論の失敗。Agent の責任なので、有効な試行として `error_rate` に数える。
INFERENCE_FAILURE_KINDS = ("agent_error", "budget")
#: インフラの失敗（GPU マシン）。新しいセッションで同じ試行をやり直す。
GPU_INFRA_KINDS = ("infrastructure", "capacity_full", "env_startup")
#: インフラの失敗（LLM サーバ）。同じセッションで同じ試行をやり直す。`llm_unreachable` は
#: 一瞬の接続リセットでも出るのでやり直すが、使い切ったらスイートを止める（`suite.py`）。
LLM_INFRA_KINDS = ("llm_timeout", "llm_unreachable")
#: 設定の失敗。続けても全部失敗するので、スイートを止める。Agent の読み込みの失敗と、
#: セッションを作るときの 4xx（429 以外）も同じ扱いで、`suite.py` が見分ける。
CONFIG_KINDS = ("llm_auth",)

INFRA_EVENTS = "infra_events.jsonl"

#: `summary.json` の `notes` に残す注意。robosuite は seed が初期配置に届かない。
ROBOSUITE_SEED_NOTE = (
    "robosuite のタスク（{tasks}）は seed が初期配置に効かない。"
    "同じ seed でも試行ごとに配置が変わる"
)


def summarize(run_dir: str | Path) -> dict[str, Any]:
    """実行フォルダを読んで `summary.json` の中身を返す。書き出しはしない。"""
    run_dir = Path(run_dir)
    suite = yaml.safe_load((run_dir / "suite.yaml").read_text(encoding="utf-8"))
    run = _read_json(run_dir / "run.json") or {}

    per_task = {task: _results(run_dir / task, suite["trials"]) for task in suite["tasks"]}
    tasks = {
        task: _summarize_task(run_dir / task, suite["trials"], found) for task, found in per_task.items()
    }
    scored = [t for t in tasks.values() if t["valid_trials"]]
    results = [r for found in per_task.values() for r in found.values()]

    robosuite = [t for t in suite["tasks"] if not t.startswith("libero_")]
    return {
        "schema_version": SCHEMA_VERSION,
        "suite": suite,
        "agent": run.get("agent"),
        "git": run.get("git"),
        "started_at": run.get("started_at"),
        "finished_at": run.get("finished_at"),
        "complete": all(not t["not_run"] for t in tasks.values()),
        "tasks": tasks,
        "macro": {
            "tasks": len(scored),
            "pass_at_1": _mean([t["pass_at_1"] for t in scored]),
            "error_rate": _mean([t["error_rate"] for t in scored]),
            "wall_clock_s_median": _mean([t["wall_clock_s"]["median"] for t in scored]),
        },
        "micro": {
            "valid_trials": len(results),
            "pass_at_1": _rate(sum(r.get("task_completed") is True for r in results), len(results)),
            "error_rate": _rate(sum(bool(_inference_errors(r)) for r in results), len(results)),
            "wall_clock_s_median": _stats([r["wall_clock_s"] for r in results])["median"],
        },
        "infra": _total_infra(tasks.values()),
        "notes": [ROBOSUITE_SEED_NOTE.format(tasks=", ".join(robosuite))] if robosuite else [],
    }


def write_summary(run_dir: str | Path) -> dict[str, Any]:
    """集計して `summary.json` に書き、中身を返す。"""
    summary = summarize(run_dir)
    (Path(run_dir) / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def render(summary: dict[str, Any]) -> str:
    """端末に出す表。"""
    suite = summary["suite"]
    agent = summary.get("agent") or {}
    lines = [
        f"suite {suite['name']}  ({agent.get('spec')}, model={agent.get('model') or '未設定'},"
        f" max_steps={suite['max_steps']}, trials={suite['trials']})",
        "",
    ]
    rows = [("task", "pass@1", "error_rate", "wall_clock 中央値", "備考")]
    for name, t in summary["tasks"].items():
        valid = t["valid_trials"]
        rows.append((
            name,
            f"{t['completed']}/{valid}",
            f"{t['errored']}/{valid}",
            _seconds(t["wall_clock_s"]["median"]),
            _remarks(t),
        ))
    macro = summary["macro"]
    total = (
        f"macro ({macro['tasks']})",
        _ratio(macro["pass_at_1"]),
        _ratio(macro["error_rate"]),
        _seconds(macro["wall_clock_s_median"]),
        "",
    )
    width = [max(_display_width(r[i]) for r in (*rows, total)) for i in range(4)]
    table = [_row(r, width) for r in rows]
    table.append("-" * max(_display_width(line) for line in table))
    table.append(_row(total, width))
    lines.extend(table)

    infra = summary["infra"]
    failed = ", ".join(f"{name} trial {n}" for name, t in summary["tasks"].items() for n in t["infra"]["failed"])
    lines.append("")
    lines.append(
        f"infra: 失敗 {infra['failed_trials']}/{infra['attempted_trials']} 試行"
        + (f"（{failed}）" if failed else "")
        + f" / やり直し GPU {infra['retries']['gpu']} 回, LLM {infra['retries']['llm']} 回"
    )
    not_run = sum(len(t["not_run"]) for t in summary["tasks"].values())
    if not summary["complete"]:
        lines.append(f"未完了: 途中で止まったので {not_run} 試行を回していない。指標は回した試行だけのもの")
    lines.extend(f"注意: {note}" for note in summary["notes"])
    return "\n".join(lines)


def trial_dirs(task_dir: Path) -> dict[int, Path]:
    """`result.json` がある試行フォルダ。試行番号 → フォルダ。

    `_discarded/` の下は見ない（1 階層しか探さない）。
    """
    found: dict[int, Path] = {}
    for result in sorted(task_dir.glob("trial_*/result.json"), key=lambda p: p.stat().st_mtime):
        number = _trial_number(result.parent.name)
        if number is not None:
            found[number] = result.parent
    return found


def read_infra_events(task_dir: Path) -> list[dict[str, Any]]:
    path = task_dir / INFRA_EVENTS
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# -- 内部 --------------------------------------------------------------


def _summarize_task(task_dir: Path, trials: int, results: dict[int, dict[str, Any]]) -> dict[str, Any]:
    events = [e for e in read_infra_events(task_dir) if 1 <= e["trial"] <= trials]
    gave_up = {e["trial"] for e in events if e.get("gave_up")}
    failed = sorted(gave_up - results.keys())
    not_run = [n for n in range(1, trials + 1) if n not in results and n not in gave_up]

    valid = list(results.values())
    errors = {"step_error": 0, **dict.fromkeys(INFERENCE_FAILURE_KINDS, 0)}
    errored = 0
    for r in valid:
        found = _inference_errors(r)
        errored += bool(found)
        for kind in found:
            errors[kind] += 1
    completed = sum(r.get("task_completed") is True for r in valid)
    attempted = len(valid) + len(failed)

    return {
        "trials": trials,
        "valid_trials": len(valid),
        "completed": completed,
        "errored": errored,
        "unscored": sum(r.get("task_completed") is None for r in valid),
        "not_run": not_run,
        "pass_at_1": _rate(completed, len(valid)),
        "error_rate": _rate(errored, len(valid)),
        "inference_errors": errors,
        "infra": {
            "failed": failed,
            "failure_rate": _rate(len(failed), attempted),
            "retries": _count_retries(events),
        },
        "wall_clock_s": _stats([r["wall_clock_s"] for r in valid]),
        "execution_time_s": _stats([r["execution_time_used_s"] for r in valid]),
        "seeds": {str(n): results[n].get("seed") for n in sorted(results)},
    }


def _results(task_dir: Path, trials: int) -> dict[int, dict[str, Any]]:
    return {
        n: json.loads((d / "result.json").read_text(encoding="utf-8"))
        for n, d in trial_dirs(task_dir).items()
        if 1 <= n <= trials
    }


def _inference_errors(result: dict[str, Any]) -> list[str]:
    """この試行で起きた推論の失敗の内訳。何も無ければ空。"""
    found = []
    if any(not s.get("ok", True) for s in result.get("steps") or []):
        found.append("step_error")
    kind = (result.get("failure") or {}).get("kind")
    if kind in INFERENCE_FAILURE_KINDS:
        found.append(kind)
    return found


def _count_retries(events: list[dict[str, Any]]) -> dict[str, int]:
    """やり直した回数。使い切ったあとの最後の失敗は「やり直し」に数えない。"""
    counts = {"gpu": 0, "llm": 0}
    for e in events:
        if not e.get("gave_up"):
            counts[e["category"]] = counts.get(e["category"], 0) + 1
    return counts


def _total_infra(tasks) -> dict[str, Any]:
    tasks = list(tasks)
    failed = sum(len(t["infra"]["failed"]) for t in tasks)
    attempted = sum(t["valid_trials"] + len(t["infra"]["failed"]) for t in tasks)
    return {
        "failed_trials": failed,
        "attempted_trials": attempted,
        "failure_rate": _rate(failed, attempted),
        "retries": {
            "gpu": sum(t["infra"]["retries"]["gpu"] for t in tasks),
            "llm": sum(t["infra"]["retries"]["llm"] for t in tasks),
        },
    }


def _trial_number(name: str) -> int | None:
    # `trial_01_sandboxrc_...`（`capx.utils.launch_utils._save_trial_artifacts` が付ける名前）
    parts = name.split("_")
    if len(parts) < 2 or parts[0] != "trial" or not parts[1].isdigit():
        return None
    return int(parts[1])


def _stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "p90": None}
    return {
        "mean": round(statistics.fmean(values), 3),
        "median": round(statistics.median(values), 3),
        "p90": round(_percentile(values, 0.9), 3),
    }


def _percentile(values: list[float], q: float) -> float:
    """線形補間（numpy の既定と同じ）。numpy を入れずに済ませる。"""
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


def _rate(n: int, d: int) -> float | None:
    return round(n / d, 4) if d else None


def _mean(values: list[float | None]) -> float | None:
    values = [v for v in values if v is not None]
    return round(statistics.fmean(values), 4) if values else None


def _read_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _row(cells: tuple[str, ...], width: list[int]) -> str:
    padded = [cell + " " * (width[i] - _display_width(cell)) for i, cell in enumerate(cells[:4])]
    return "  ".join([*padded, cells[4]]).rstrip()


def _display_width(text: str) -> int:
    """端末での幅。全角（「中央値」など）は 2 文字分。"""
    import unicodedata

    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)


def _seconds(value: float | None) -> str:
    return "-" if value is None else f"{value:.1f}s"


def _ratio(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def _remarks(task: dict[str, Any]) -> str:
    parts = [f"{k} {v}" for k, v in task["inference_errors"].items() if v]
    if task["unscored"]:
        parts.append(f"unscored {task['unscored']}")
    if task["infra"]["failed"]:
        parts.append(f"infra_failed {len(task['infra']['failed'])}")
    if task["not_run"]:
        parts.append(f"未実行 {len(task['not_run'])}")
    return ", ".join(parts)


def main(run_dir: str, /) -> None:
    """実行フォルダを集計し直して `summary.json` を書き、表を出す。

    Args:
        run_dir: `suite.py` が作った実行フォルダ（`suite.yaml` がある場所）。
    """
    print(render(write_summary(run_dir)))


if __name__ == "__main__":
    tyro.cli(main)
