"""録画フレームのバッファ。上限を超えたら間引いて、試行の最後まで動画を残す。

シミュレータに依存しない（robosuite を import しない）ので、単体でテストできる。
ホストのクラスは `_frame_buffer` / `_wrist_frame_buffer` / `_subsample_rate` を
持ち、フレームを録るたびに `_append_frame()` を呼ぶ。
"""

from __future__ import annotations

import numpy as np


class FrameBufferMixin:
    _SUBSAMPLE_RATE: int = 5

    _frame_buffer: list[np.ndarray]
    _wrist_frame_buffer: list[np.ndarray]
    _subsample_rate: int

    def _init_frame_buffer(self) -> None:
        self._frame_buffer = []
        self._wrist_frame_buffer = []
        self._subsample_rate = self._SUBSAMPLE_RATE
        # フレームの上限（バイト）。None なら無制限（ローカル実行の既定）。
        # 超えたら古いフレームを 1 つ飛ばしに捨て、以後の録画間隔を倍にして、
        # 試行の最後まで動画を残す。worker が設定する。
        self._frame_budget_bytes: int | None = None
        # 各フレームの通し番号。間引いてもターンの範囲指定がずれないようにする。
        self._frame_seqs: list[int] = []
        self._frame_seq_next = 0
        self._pinned_seqs: set[int] = set()
        self._frame_stats: dict = {"recorded": 0, "dropped": 0, "rate_changes": []}

    def _append_frame(self, frame: np.ndarray, wrist: np.ndarray | None = None) -> None:
        self._sync_seqs()
        self._frame_buffer.append(frame)
        self._frame_seqs.append(self._frame_seq_next)
        self._frame_seq_next += 1
        self._frame_stats["recorded"] += 1
        if wrist is not None:
            self._wrist_frame_buffer.append(wrist)
        self._thin_frames()

    def _reset_frame_recording(self) -> None:
        self._clear_frames()
        self._subsample_rate = self._SUBSAMPLE_RATE
        self._frame_stats = {"recorded": 0, "dropped": 0, "rate_changes": []}

    #
    # `get_video_frame_count()` は「これまでに録ったフレームの通し番号」を返し、
    # `get_video_frames_range(start, end)` はその番号で範囲を指定する。間引きが
    # 起きなければ番号と添字は一致するので、従来の呼び出し（ターンの前後で
    # count を取り、範囲で動画を取る）はそのまま動く。

    def set_frame_budget(self, budget_bytes: int | None) -> None:
        """フレーム（メイン + 手首）の合計バイト数の上限。None で無制限。"""
        self._frame_budget_bytes = budget_bytes

    def get_frame_stats(self) -> dict:
        """録画の記録。録った枚数・捨てた枚数・残った枚数・間隔の変更履歴。"""
        retained = len(self._frame_buffer)
        return {
            "recorded": self._frame_stats["recorded"],
            "dropped": self._frame_stats["dropped"],
            "retained": retained,
            "subsample_rate": self._subsample_rate,
            "rate_changes": list(self._frame_stats["rate_changes"]),
        }

    def _clear_frames(self) -> None:
        self._frame_buffer.clear()
        self._wrist_frame_buffer.clear()
        self._frame_seqs.clear()
        self._frame_seq_next = 0
        self._pinned_seqs.clear()

    def _sync_seqs(self) -> None:
        """サブクラスが `_frame_buffer` を直接作り直しても、番号を合わせ直す。"""
        if len(self._frame_seqs) == len(self._frame_buffer):
            return
        if not self._frame_buffer:
            self._clear_frames()
        else:
            self._frame_seqs = list(range(len(self._frame_buffer)))
            self._frame_seq_next = len(self._frame_buffer)

    def _frames_in(self, buffer: list[np.ndarray], start: int, end: int) -> list[np.ndarray]:
        self._sync_seqs()
        if len(buffer) != len(self._frame_seqs):  # 手首カメラを録っていない
            return []
        return [f.copy() for seq, f in zip(self._frame_seqs, buffer) if start <= seq < end]

    def get_video_frames(self, *, clear: bool = False) -> list[np.ndarray]:
        if clear:
            # 取り出すだけなので複製しない（複製すると一時的に 2 倍のメモリを使う）
            frames = self._frame_buffer
            self._frame_buffer = []
            self._wrist_frame_buffer.clear()
            self._frame_seqs.clear()
            self._pinned_seqs.clear()
            self._frame_seq_next = 0
            return frames
        return [frame.copy() for frame in self._frame_buffer]

    def get_video_frame_count(self) -> int:
        """次に録るフレームの通し番号。ターンの境界のフレームを間引きから守る。"""
        self._sync_seqs()
        n = self._frame_seq_next
        # そのターンの最初のフレーム（n）と、前のターンの最後のフレーム（n-1）
        self._pinned_seqs.update((n - 1, n))
        return n

    def get_video_frames_range(self, start: int, end: int) -> list[np.ndarray]:
        return self._frames_in(self._frame_buffer, start, end)

    def get_wrist_video_frames(self, *, clear: bool = False) -> list[np.ndarray]:
        frames = [frame.copy() for frame in self._wrist_frame_buffer]
        if clear:
            self._wrist_frame_buffer.clear()
        return frames

    def get_wrist_video_frames_range(self, start: int, end: int) -> list[np.ndarray]:
        return self._frames_in(self._wrist_frame_buffer, start, end)

    def _buffered_bytes(self) -> int:
        return sum(f.nbytes for f in self._frame_buffer) + sum(
            f.nbytes for f in self._wrist_frame_buffer
        )

    def _thin_frames(self) -> None:
        """上限を超えたら、1 つ飛ばしに捨てて、以後の録画間隔を倍にする。

        最新のフレーム・最初のフレーム・ターンの境界のフレームは残す。
        残すべきものだけで上限を超える場合は、それ以上は捨てない。
        """
        budget = self._frame_budget_bytes
        if budget is None:
            return
        while self._buffered_bytes() > budget and len(self._frame_buffer) > 2:
            n = len(self._frame_buffer)
            keep = [
                i % 2 == 0 or i == n - 1 or self._frame_seqs[i] in self._pinned_seqs
                for i in range(n)
            ]
            if all(keep):
                break
            has_wrist = len(self._wrist_frame_buffer) == n
            self._frame_stats["dropped"] += keep.count(False)
            self._frame_buffer = [f for f, k in zip(self._frame_buffer, keep) if k]
            if has_wrist:
                self._wrist_frame_buffer = [
                    f for f, k in zip(self._wrist_frame_buffer, keep) if k
                ]
            self._frame_seqs = [s for s, k in zip(self._frame_seqs, keep) if k]
            self._subsample_rate *= 2
            self._frame_stats["rate_changes"].append(
                {"at_seq": self._frame_seq_next, "subsample_rate": self._subsample_rate}
            )
