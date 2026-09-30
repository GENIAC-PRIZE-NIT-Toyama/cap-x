"""doctor は、繋がらない相手にも例外を投げず、原因と直し方を返す。"""

from __future__ import annotations

from capx.bench import doctor


def test_missing_server_url_says_how_to_fix() -> None:
    (check,) = doctor.check_server(None)
    assert not check.ok and ".env" in check.fix


def test_unreachable_server_is_reported_not_raised() -> None:
    (check,) = doctor.check_server("http://127.0.0.1:9")  # 誰も待っていないポート
    assert not check.ok and "届かない" in check.detail


def test_missing_agent_file_is_reported() -> None:
    assert not doctor.check_agent("/no/such/agent.py").ok


def test_the_example_agent_loads() -> None:
    from pathlib import Path

    example = Path(__file__).resolve().parent.parent / "remote/agents/example_agent.py"
    assert doctor.check_agent(str(example)).ok


def test_render_shows_the_fix_only_for_failures() -> None:
    text = doctor.render(
        [doctor.Check("A", True, "ok"), doctor.Check("B", False, "bad", "do this")]
    )
    assert "do this" in text and "1 件" in text
