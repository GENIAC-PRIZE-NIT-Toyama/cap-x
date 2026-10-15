"""点群のノイズ除去。DBSCAN と同じ判定を、少ないメモリで計算する。

LIBERO の API（`capx/integrations/franka/libero.py` / `libero_reduced.py`）の
`filter_noise` は、scikit-learn の DBSCAN の結果から「ノイズかどうか」だけを使う。
DBSCAN は全ての点の近傍の一覧を一度にメモリに並べるので、物体に近づいて点が密に
なると、点の数の 2 乗に近い勢いでメモリを使う（24 万点で +2.6GB）。worker の
メモリ上限を超えて強制終了されていた。

ここでは同じ定義で、近傍を「数えるだけ」にして計算する。

- 中心点: 半径 eps 以内に、自分を含めて min_samples 点以上ある点
- ノイズ: 中心点でなく、半径 eps 以内に中心点が 1 つも無い点

点を `chunk` 点ずつ区切って数えるので、メモリは点の数に比例する程度で済む
（24 万点で +36MB、時間は DBSCAN の半分以下。合成データでの測定）。点を間引かないので、
後段（GraspNet）に渡す点群は DBSCAN を使ったときと同じになる。
"""

from __future__ import annotations

import os

import numpy as np

#: 調査用。`CAPX_CHECK_NOISE_FILTER=1` のとき、scikit-learn の DBSCAN でも計算して、
#: 判定が違う点の数を出す（実データで同じ結果になるかを確かめるため）。
CHECK_AGAINST_DBSCAN = os.environ.get("CAPX_CHECK_NOISE_FILTER") == "1"


def dbscan_keep_mask(
    points: np.ndarray, eps: float, min_samples: int, *, chunk: int = 20_000
) -> np.ndarray:
    """DBSCAN でノイズにならない点を True にした配列を返す（`labels != -1` と同じ）。

    Args:
        points: (N, 3) の点群。
        eps: 近傍とみなす距離（この距離ちょうども含む。DBSCAN と同じ）。
        min_samples: 中心点になるのに要る、半径 eps 内の点の数（自分を含む）。
        chunk: 一度に近傍を数える点の数。メモリと速さの兼ね合い。
    """
    from scipy.spatial import cKDTree

    points = np.asarray(points, dtype=np.float64)
    n = len(points)
    if n == 0:
        return np.zeros(0, dtype=bool)

    tree = cKDTree(points)
    counts = np.empty(n, dtype=np.int64)
    for start in range(0, n, chunk):
        counts[start : start + chunk] = tree.query_ball_point(
            points[start : start + chunk], r=eps, return_length=True
        )
    core = counts >= min_samples
    keep = core.copy()

    # 中心点でない点は、近くに中心点があればノイズではない（DBSCAN の境界点）
    rest = np.flatnonzero(~core)
    if core.any() and len(rest):
        core_tree = cKDTree(points[core])
        for start in range(0, len(rest), chunk):
            idx = rest[start : start + chunk]
            keep[idx] = core_tree.query_ball_point(points[idx], r=eps, return_length=True) > 0

    if CHECK_AGAINST_DBSCAN:
        _report_difference(points, eps, min_samples, keep)
    return keep


def _report_difference(points: np.ndarray, eps: float, min_samples: int, keep: np.ndarray) -> None:
    from sklearn.cluster import DBSCAN

    expected = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(points) != -1
    diff = int((expected != keep).sum())
    print(
        f"[noise-filter check] points={len(points)} diff={diff} "
        f"noise_kdtree={int((~keep).sum())} noise_dbscan={int((~expected).sum())}",
        flush=True,
    )
