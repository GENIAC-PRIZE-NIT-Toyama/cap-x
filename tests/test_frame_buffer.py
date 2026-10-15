"""録画フレームのバッファ。JPEG で持ち、上限を超えたら間引く。

見たいのは 4 点。溜めたものが JPEG で、取り出すと元の画像の大きさに戻ること。
上限を超えても試行の最後まで動画が残ること。間引いてもターンの範囲指定（前後で
count を取って範囲で動画を取る）がずれないこと。1 枚ずつの取り出しが、まとめての
取り出しと同じものを返すこと。

画像は単色にしてある。JPEG は単色をほぼそのまま戻すが、画素値は ±数 ずれうるので、
値は誤差を見込んで比べる。
"""

from __future__ import annotations

import numpy as np

from capx.utils.frame_buffer import FrameBufferMixin, encode_frame

FRAME = 64
#: JPEG 1 枚の大きさ（単色なので色によらずほぼ同じ）。上限はこの何枚分かで決める。
ONE = len(encode_frame(np.full((FRAME, FRAME, 3), 128, dtype=np.uint8)))
TOLERANCE = 3


class _Env(FrameBufferMixin):
    """シミュレータの録画規則を真似る。`step % 間引き間隔 == 0` のときだけ録る。

    フレームの画素値は `step * scale`（256 で割った余り）。どのステップの
    フレームかを、取り出したあとに値から見分けるため。
    """

    _SUBSAMPLE_RATE = 1

    def __init__(self, budget: int | None = None, *, scale: int = 1, wrist: bool = False) -> None:
        self._init_frame_buffer()
        self.set_frame_budget(budget)
        self.step = 0
        self.scale = scale
        self.wrist = wrist

    def run(self, n_steps: int) -> None:
        for _ in range(n_steps):
            if self.step % self._subsample_rate == 0:
                frame = np.full((FRAME, FRAME, 3), (self.step * self.scale) % 256, dtype=np.uint8)
                self._append_frame(frame, frame.copy() if self.wrist else None)
            self.step += 1


def _values(frames) -> list[int]:
    return [int(round(float(f.mean()))) for f in frames]


def _near(a: int, b: int) -> bool:
    return abs(a - b) <= TOLERANCE


def test_frames_are_kept_as_jpeg_and_come_back_as_images() -> None:
    env = _Env(None, scale=2)
    env.run(10)
    assert all(isinstance(f, bytes) and f[:2] == b"\xff\xd8" for f in env._frame_buffer)
    frames = env.get_video_frames()
    assert all(f.shape == (FRAME, FRAME, 3) and f.dtype == np.uint8 for f in frames)
    assert all(_near(v, 2 * i) for i, v in enumerate(_values(frames)))


def test_without_a_budget_nothing_is_dropped() -> None:
    env = _Env(None)
    env.run(200)
    assert len(env.get_video_frames()) == 200
    assert env.get_frame_stats()["dropped"] == 0


def test_over_budget_keeps_the_whole_trial_at_a_coarser_rate() -> None:
    """上限の 20 枚分を大きく超えても、最初から最後まで残る。"""
    budget = 20 * ONE
    env = _Env(budget=budget)
    env.run(200)

    kept = _values(env.get_video_frames())
    assert env._buffered_bytes() <= budget + 2 * ONE  # 単色でも色ごとに数バイト違う
    assert _near(kept[0], 0), "最初が残る"
    assert kept[-1] >= 190 - TOLERANCE, "最後が残る"
    assert max(b - a for a, b in zip(kept, kept[1:])) < 200 // 3, "途中が欠けていない"

    stats = env.get_frame_stats()
    assert stats["recorded"] < 200, "間隔が広がった分、録る枚数も減る"
    assert stats["dropped"] + stats["retained"] == stats["recorded"]
    assert stats["subsample_rate"] > env._SUBSAMPLE_RATE
    assert stats["rate_changes"], "間隔を変えた履歴が残る"


def test_turn_ranges_still_point_at_the_right_frames_after_thinning() -> None:
    """ターンの前後で count を取り、その範囲で動画を取る使い方が、間引き後も正しい。"""
    env = _Env(budget=20 * ONE)
    env.run(100)

    start = env.get_video_frame_count()  # 次のターンの開始
    step_start = env.step
    env.run(30)
    end = env.get_video_frame_count()
    step_end = env.step

    env.run(300)  # その後さらに間引かれる

    turn = _values(env.get_video_frames_range(start, end))
    assert turn, "そのターンのフレームが残っている"
    assert all(step_start - TOLERANCE <= v < step_end + TOLERANCE for v in turn), "別のターンが混じらない"
    # ターンで最初に録られたフレーム（通し番号 start）は間引きから守られる。間引きのあとは
    # 録画の間隔が広がっているので、それがステップ step_start ちょうどとは限らない。
    assert start in env._frame_seqs, "ターンの最初のフレームは間引きから守られる"
    assert turn[0] == _values(env.get_video_frames_range(start, start + 1))[0]


def test_range_matches_indices_when_nothing_was_dropped() -> None:
    env = _Env(None, scale=20)
    env.run(10)
    assert [_near(v, e) for v, e in zip(_values(env.get_video_frames_range(3, 7)), (60, 80, 100, 120))] == [True] * 4
    assert env.get_video_frame_count() == 10


def test_iterating_a_range_gives_the_same_frames_one_at_a_time() -> None:
    env = _Env(None, scale=20)
    env.run(10)
    one_by_one = env.iter_video_frames_range(2, 8)
    assert not isinstance(one_by_one, list), "まとめて戻さない"
    assert [f.tolist() for f in one_by_one] == [f.tolist() for f in env.get_video_frames_range(2, 8)]


def test_taking_the_frames_out_empties_the_buffer() -> None:
    env = _Env(None)
    env.run(5)
    assert len(env.get_video_frames(clear=True)) == 5
    assert env.get_video_frame_count() == 0
    assert env._buffered_bytes() == 0
    env.run(1)
    assert len(env.get_video_frames()) == 1


def test_taking_the_main_frames_out_leaves_the_wrist_frames() -> None:
    """メインを取り出したあとでも、手首の動画は別に取り出せる（保存でこの順に呼ぶ）。"""
    env = _Env(None, wrist=True)
    env.run(5)
    assert len(env.get_video_frames(clear=True)) == 5
    assert len(env.get_wrist_video_frames(clear=True)) == 5, "手首が一緒に消えない"
    assert env._buffered_bytes() == 0
    assert env.get_video_frame_count() == 0


def test_wrist_frames_are_thinned_together_with_the_main_frames() -> None:
    env = _Env(budget=40 * ONE, wrist=True)
    env.run(100)
    main = _values(env.get_video_frames())
    wrist = _values(env.get_wrist_video_frames())
    assert main == wrist
    assert env._buffered_bytes() <= 40 * ONE + 4 * ONE


class _CodeEnvWithFrames:
    """`CodeExecutionEnvBase` の代役。step のたびにフレームを 12 枚録る。

    まとめての取り出し（`get_video_frames_range`）は使わせない。ターン動画が
    1 枚ずつの取り出しで作られていることを確かめるため。
    """

    def __init__(self) -> None:
        from types import SimpleNamespace

        self.buffer = _Env(None, scale=10)
        self.low_level_env = SimpleNamespace(_sim_step_count=0, render_camera_names=("main",))

    def reset(self, *, seed=None, options=None):
        return {}, {"task_prompt": "Goal: x"}

    def step(self, code):
        self.buffer.run(12)
        info = {"sandbox_rc": 0, "stdout": "", "stderr": "", "task_completed": False}
        return {}, 0.0, False, False, info

    def get_video_frame_count(self) -> int:
        return self.buffer.get_video_frame_count()

    def iter_video_frames_range(self, start, end):
        return self.buffer.iter_video_frames_range(start, end)

    def get_video_frames_range(self, start, end):
        raise AssertionError("まとめて取り出してはいけない")


def test_the_turn_video_is_written_one_frame_at_a_time(tmp_path) -> None:
    import imageio.v3 as iio

    from capx.local_env import LocalAgentEnv

    env = LocalAgentEnv(_CodeEnvWithFrames(), task_id="fake")
    env.reset_for_trial(1, 1)
    result = env.step("move()", capture_video=True)

    assert result.video and result.video_media_type == "video/mp4"
    path = tmp_path / "turn.mp4"
    path.write_bytes(result.video)
    assert iio.imread(path).shape[0] == 12, "そのターンの 12 枚がすべて入る"
