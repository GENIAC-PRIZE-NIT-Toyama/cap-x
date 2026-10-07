"""録画フレームの間引き。

見たいのは 2 点。上限を超えても試行の最後まで動画が残ること、間引いても
ターンの範囲指定（前後で count を取って範囲で動画を取る）がずれないこと。
"""

from __future__ import annotations

import numpy as np

from capx.utils.frame_buffer import FrameBufferMixin

FRAME = 100  # 100x100x3 = 30000 bytes


class _Env(FrameBufferMixin):
    """シミュレータの録画規則を真似る。`step % 間引き間隔 == 0` のときだけ録る。"""

    _SUBSAMPLE_RATE = 1

    def __init__(self, budget: int | None = None) -> None:
        self._init_frame_buffer()
        self.set_frame_budget(budget)
        self.step = 0

    def run(self, n_steps: int) -> None:
        for _ in range(n_steps):
            if self.step % self._subsample_rate == 0:
                self._append_frame(np.full((FRAME, FRAME, 3), self.step % 256, dtype=np.uint8))
            self.step += 1


def _values(frames) -> list[int]:
    return [int(f[0, 0, 0]) for f in frames]


def test_without_a_budget_nothing_is_dropped() -> None:
    env = _Env(None)
    env.run(200)
    assert len(env.get_video_frames()) == 200
    assert env.get_frame_stats()["dropped"] == 0


def test_over_budget_keeps_the_whole_trial_at_a_coarser_rate() -> None:
    """予算の 20 枚分を大きく超えても、最初から最後まで残る。"""
    env = _Env(budget=20 * FRAME * FRAME * 3)
    env.run(200)

    kept = _values(env.get_video_frames())
    assert env._buffered_bytes() <= 20 * FRAME * FRAME * 3
    assert kept[0] == 0, "最初が残る"
    assert kept[-1] >= 190, "最後が残る"
    # 途中が欠けていない（間隔は粗くなるが、全体に散らばる）
    assert max(b - a for a, b in zip(kept, kept[1:])) < 200 // 3

    stats = env.get_frame_stats()
    assert stats["recorded"] < 200, "間隔が広がった分、録る枚数も減る"
    assert stats["dropped"] + stats["retained"] == stats["recorded"]
    assert stats["subsample_rate"] > env._SUBSAMPLE_RATE
    assert stats["rate_changes"], "間隔を変えた履歴が残る"


def test_turn_ranges_still_point_at_the_right_frames_after_thinning() -> None:
    """ターンの前後で count を取り、その範囲で動画を取る使い方が、間引き後も正しい。"""
    env = _Env(budget=20 * FRAME * FRAME * 3)
    env.run(100)

    start = env.get_video_frame_count()  # 次のターンの開始
    step_start = env.step
    env.run(30)
    end = env.get_video_frame_count()
    step_end = env.step

    env.run(300)  # その後さらに間引かれる

    turn = _values(env.get_video_frames_range(start, end))
    assert turn, "そのターンのフレームが残っている"
    assert all(step_start <= v < step_end for v in turn), "別のターンのフレームが混じらない"
    assert turn[0] == step_start % 256 or turn[0] >= step_start, "ターンの先頭付近が残る"


def test_range_matches_indices_when_nothing_was_dropped() -> None:
    env = _Env(None)
    env.run(10)
    assert _values(env.get_video_frames_range(3, 7)) == [3, 4, 5, 6]
    assert env.get_video_frame_count() == 10


def test_taking_the_frames_out_empties_the_buffer() -> None:
    env = _Env(None)
    env.run(5)
    assert len(env.get_video_frames(clear=True)) == 5
    assert env.get_video_frame_count() == 0
    env.run(1)
    assert len(env.get_video_frames()) == 1


def test_wrist_frames_are_thinned_together_with_the_main_frames() -> None:
    env = _Env(budget=20 * FRAME * FRAME * 3)
    for i in range(100):
        if i % env._subsample_rate == 0:
            frame = np.full((FRAME, FRAME, 3), i, dtype=np.uint8)
            env._append_frame(frame, frame.copy())
    main = _values(env.get_video_frames())
    wrist = _values(env._wrist_frame_buffer)
    assert main == wrist
    assert env._buffered_bytes() <= 20 * FRAME * FRAME * 3
