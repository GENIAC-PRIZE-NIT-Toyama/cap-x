"""点群のノイズ除去が、scikit-learn の DBSCAN とまったく同じ判定になること。

LIBERO の API の `filter_noise` は、DBSCAN の代わりにこれを使う。後段の GraspNet は、
物体の上の把持を「接触点が物体の点群の 0.1mm 以内」で選ぶので、残す点が 1 点でも違うと
結果が変わりうる。だから「ほぼ同じ」ではなく、1 点も違わないことを確かめる。

scipy と scikit-learn が要る（worker の環境にはある。参加者の軽い環境では skip）。
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("scipy")
DBSCAN = pytest.importorskip("sklearn.cluster").DBSCAN

from capx.utils.noise_filter import dbscan_keep_mask  # noqa: E402

EPS, MIN = 0.005, 10  # LIBERO の filter_noise と同じ


def _expected(points: np.ndarray) -> np.ndarray:
    return DBSCAN(eps=EPS, min_samples=MIN).fit_predict(points) != -1


def _bowl(n: int, seed: int = 0) -> np.ndarray:
    """お椀ほどの半球の表面に n 点（深度の誤差 0.5mm）と、2% の外れ点。"""
    rng = np.random.default_rng(seed)
    u, v = rng.random(n), rng.random(n)
    theta, phi = 2 * np.pi * u, np.arccos(1 - v)
    r = 0.06 + rng.normal(0, 0.0005, n)
    surface = np.stack(
        [r * np.sin(phi) * np.cos(theta), r * np.sin(phi) * np.sin(theta), r * np.cos(phi)], 1
    )
    return np.concatenate([surface, rng.uniform(-0.08, 0.08, size=(n // 50, 3))])


def _depth_like_grid(seed: int = 0) -> np.ndarray:
    """深度画像の画素から作った点群のように、格子状に並んだ点。

    間隔 2.5mm なので、2 つ隣の点がちょうど 5mm（境界）になる。境界の扱いの違いは、
    ここで一番出やすい。まばらな外れ点も混ぜる。
    """
    rng = np.random.default_rng(seed)
    ys, xs = np.mgrid[0:120, 0:160]
    z = 0.4 + 0.0005 * np.sin(xs / 7.0)  # なだらかな面
    grid = np.stack([xs * 0.0025, ys * 0.0025, z], -1).reshape(-1, 3)
    return np.concatenate([grid, rng.uniform(0, 0.4, size=(300, 3))])


@pytest.mark.parametrize(
    "points",
    [
        pytest.param(_bowl(3_000), id="bowl-3k"),
        pytest.param(_bowl(30_000, seed=1), id="bowl-30k"),
        pytest.param(_depth_like_grid(), id="depth-like-grid"),
        pytest.param(np.random.default_rng(2).uniform(0, 0.05, size=(500, 3)), id="sparse-noise"),
    ],
)
def test_same_noise_points_as_dbscan(points) -> None:
    keep = dbscan_keep_mask(points, eps=EPS, min_samples=MIN)
    expected = _expected(points)
    assert keep.dtype == bool and keep.shape == expected.shape
    assert int((keep != expected).sum()) == 0


def test_small_and_duplicated_clouds() -> None:
    few = np.zeros((5, 3))  # 同じ位置に 5 点だけ（10 点に届かない）
    assert not dbscan_keep_mask(few, eps=EPS, min_samples=MIN).any()
    assert (dbscan_keep_mask(few, eps=EPS, min_samples=MIN) == _expected(few)).all()

    stacked = np.zeros((12, 3))  # 同じ位置に 12 点（重複も 1 点ずつ数える）
    assert (dbscan_keep_mask(stacked, eps=EPS, min_samples=MIN) == _expected(stacked)).all()


def test_empty_cloud() -> None:
    assert dbscan_keep_mask(np.zeros((0, 3)), eps=EPS, min_samples=MIN).shape == (0,)


def test_chunking_does_not_change_the_result() -> None:
    points = _bowl(20_000, seed=3)
    whole = dbscan_keep_mask(points, eps=EPS, min_samples=MIN, chunk=10**9)
    pieces = dbscan_keep_mask(points, eps=EPS, min_samples=MIN, chunk=777)
    assert (whole == pieces).all()
