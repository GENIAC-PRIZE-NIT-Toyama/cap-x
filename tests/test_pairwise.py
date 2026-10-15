"""全体カメラと手首カメラの点群が、1cm 未満に近づいているかの判定。

元のコードは全ての組の距離を一度に計算していて、LIBERO の worker がメモリ不足で
落ちた。区切って計算しても、判定の結果が元とまったく同じであることを確かめる。
"""

from __future__ import annotations

import numpy as np
import pytest

from capx.utils.pairwise import BYTES_PER_PAIR, any_pair_within

THRESHOLD = 0.01  # LIBERO の API と同じ 1cm


def _original(a: np.ndarray, b: np.ndarray, threshold: float) -> bool:
    """libero.py の元の式。"""
    distances = np.linalg.norm(a[:, np.newaxis, :] - b[np.newaxis, :, :], axis=2)
    return bool(np.min(distances, axis=1).min() < threshold)


def _clouds(seed: int, na: int, nb: int, gap: float):
    """a は原点付近、b は x 方向に gap だけ離れた付近に散らばる点群。"""
    rng = np.random.default_rng(seed)
    a = rng.normal(0, 0.02, size=(na, 3))
    b = rng.normal(0, 0.02, size=(nb, 3)) + np.array([gap, 0.0, 0.0])
    return a, b


@pytest.mark.parametrize("gap", [0.0, 0.05, 0.08, 0.1, 0.12, 0.2, 1.0])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_same_answer_as_the_original_expression(seed: int, gap: float) -> None:
    a, b = _clouds(seed, 600, 900, gap)
    assert any_pair_within(a, b, THRESHOLD) == _original(a, b, THRESHOLD)


def test_both_answers_are_covered() -> None:
    """上のテストが、True と False の両方を実際に通っていること。"""
    near = _clouds(0, 600, 900, 0.0)
    far = _clouds(0, 600, 900, 1.0)
    assert _original(*near, THRESHOLD) is True
    assert _original(*far, THRESHOLD) is False


def test_pairs_right_at_the_threshold() -> None:
    """閾値ちょうど・わずかに下・わずかに上の組で、元と同じ判定になる（未満で判定）。"""
    a = np.zeros((1, 3))
    for d in (THRESHOLD, np.nextafter(THRESHOLD, 0), np.nextafter(THRESHOLD, 1)):
        b = np.array([[d, 0.0, 0.0]])
        assert any_pair_within(a, b, THRESHOLD) == _original(a, b, THRESHOLD)


@pytest.mark.parametrize("max_bytes", [1, 64 * 900, 64 * 900 * 7, 2**30])
def test_the_chunk_size_does_not_change_the_answer(max_bytes: int) -> None:
    """1 行ずつから一度に全部まで、区切りの大きさで結果が変わらない。"""
    for gap in (0.0, 0.1, 0.12, 1.0):
        a, b = _clouds(5, 600, 900, gap)
        assert any_pair_within(a, b, THRESHOLD, max_bytes=max_bytes) == _original(a, b, THRESHOLD)


def test_each_chunk_stays_within_the_memory_cap(monkeypatch) -> None:
    """区切った 1 回分の途中の配列が、上限（既定 256MB）に収まる行数で計算する。"""
    seen_rows = []
    real_norm = np.linalg.norm

    def recording_norm(x, *args, **kwargs):
        seen_rows.append(x.shape[0])
        return real_norm(x, *args, **kwargs)

    monkeypatch.setattr(np.linalg, "norm", recording_norm)
    a, b = _clouds(7, 12_000, 12_000, 1.0)  # 元の式なら 12000 × 12000 × 64 バイト ≈ 8.6GB
    assert any_pair_within(a, b, THRESHOLD) is False
    assert max(seen_rows) * len(b) * BYTES_PER_PAIR <= 256 * 2**20
    assert sum(seen_rows) == len(a), "全ての点を調べている"


def test_stops_as_soon_as_a_close_pair_is_found(monkeypatch) -> None:
    calls = []
    real_norm = np.linalg.norm
    monkeypatch.setattr(np.linalg, "norm", lambda x, *a, **k: (calls.append(1), real_norm(x, *a, **k))[1])
    a, b = _clouds(8, 5_000, 5_000, 0.0)  # 最初の区切りで見つかる
    assert any_pair_within(a, b, THRESHOLD, max_bytes=64 * 5_000 * 100) is True
    assert len(calls) == 1


def test_empty_clouds() -> None:
    assert any_pair_within(np.zeros((0, 3)), np.zeros((5, 3)), THRESHOLD) is False
    assert any_pair_within(np.zeros((5, 3)), np.zeros((0, 3)), THRESHOLD) is False
