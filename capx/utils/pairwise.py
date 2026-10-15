"""2 つの点群の間の距離の判定を、メモリを抑えて行う。

LIBERO の API（`capx/integrations/franka/libero.py` の
`get_object_3d_points_and_masks_from_language`）は、全体カメラと手首カメラの物体の
点群の「全ての組」の距離を一度に計算して、1cm 未満の組があるかだけを見ていた。
途中の配列が 1 組 64 バイト（差の配列と 2 乗の配列）を使うので、1 万点 × 1.2 万点で
約 8.7GB になり、worker がメモリとスワップの上限を超えて強制終了されていた。

ここでは同じ式を、一方の点群を区切りながら計算する。各組の距離は元と同じ演算で
求めるので、判定の結果は元と変わらない。
"""

from __future__ import annotations

import numpy as np

#: 1 組の距離の計算で、途中の配列が使うバイト数。差の配列（3 成分 × 8 バイト）と
#: 2 乗の配列（同じ）、距離（8 バイト）、和の途中（8 バイト）。実測で 64.0。
BYTES_PER_PAIR = 64


def any_pair_within(
    a: np.ndarray, b: np.ndarray, threshold: float, *, max_bytes: int = 256 * 2**20
) -> bool:
    """a の点と b の点の組で、距離が threshold 未満のものが 1 組でもあるか。

    `np.min(np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2), axis=1).min() < threshold`
    と同じ結果を、途中の配列が `max_bytes` を超えないように a を区切って計算する。
    見つかった時点で打ち切る。

    Args:
        a: (N, 3) の点群。
        b: (M, 3) の点群。
        threshold: 距離の閾値（この値ちょうどは含まない）。
        max_bytes: 区切った 1 回分の計算で使う、途中の配列の上限。
    """
    a = np.asarray(a)
    b = np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return False
    rows = max(1, max_bytes // (len(b) * BYTES_PER_PAIR))
    for start in range(0, len(a), rows):
        distances = np.linalg.norm(a[start : start + rows, np.newaxis, :] - b[np.newaxis, :, :], axis=2)
        if distances.min() < threshold:
            return True
    return False
