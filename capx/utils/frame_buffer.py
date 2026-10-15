"""録画フレームのバッファ。JPEG で持ち、上限を超えたら間引いて、試行の最後まで動画を残す。

シミュレータに依存しない（robosuite を import しない）ので、単体でテストできる。
ホストのクラスは `_subsample_rate` を使って録る間隔を決め、フレームを録るたびに
`_append_frame()` を呼ぶ。

**フレームは JPEG（品質 90）で持つ。** 生の画像の配列で持つと、LIBERO（800×512、
1 枚 1.17MB）ではすぐに膨らみ、他の処理と合わせて worker のメモリ上限（4GB）を超えて
強制終了された。JPEG なら約 20 分の 1 になる（cap-x-workshop も同じ理由で同じ対策をした）。
取り出すときに画像の配列に戻すので、呼び出し側からは従来と同じに見える。
"""

from __future__ import annotations

import io
from collections.abc import Iterator

import numpy as np

#: バッファに溜める JPEG の品質。動画で見返す用途に十分で、生の約 20 分の 1。
FRAME_JPEG_QUALITY = 90


def encode_frame(frame: np.ndarray) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(frame, dtype=np.uint8)).save(
        buf, format="JPEG", quality=FRAME_JPEG_QUALITY
    )
    return buf.getvalue()


def decode_frame(data: bytes) -> np.ndarray:
    from PIL import Image

    with Image.open(io.BytesIO(data)) as image:
        return np.array(image.convert("RGB"))


class FrameBufferMixin:
    _SUBSAMPLE_RATE: int = 5

    _frame_buffer: list[bytes]
    _wrist_frame_buffer: list[bytes]
    _subsample_rate: int

    def _init_frame_buffer(self) -> None:
        self._frame_buffer = []
        self._wrist_frame_buffer = []
        self._buffer_bytes = 0
        self._subsample_rate = self._SUBSAMPLE_RATE
        # フレームの上限（JPEG のバイト数）。None なら無制限（ローカル実行の既定）。
        # 超えたら古いフレームを 1 つ飛ばしに捨て、以後の録画間隔を倍にして、
        # 試行の最後まで動画を残す。worker が設定する。
        self._frame_budget_bytes: int | None = None
        # 各フレームの通し番号。間引いてもターンの範囲指定がずれないようにする。
        self._frame_seqs: list[int] = []
        self._frame_seq_next = 0
        self._pinned_seqs: set[int] = set()
        self._frame_stats: dict = {"recorded": 0, "dropped": 0, "rate_changes": []}
        self._frame_listener = None

    def set_frame_listener(self, listener) -> None:
        """フレームを録るたびに `listener(frame)` を呼ぶ。ストリーミング用。

        渡すのは JPEG にする前の画像（ストリームは自分の品質で JPEG にする）。
        購読者がいなければ、listener の側で何もせずに返すこと（エンコード等をしない）。
        """
        self._frame_listener = listener

    def _append_frame(self, frame: np.ndarray, wrist: np.ndarray | None = None) -> None:
        self._sync_seqs()
        main = encode_frame(frame)
        self._frame_buffer.append(main)
        self._buffer_bytes += len(main)
        self._frame_seqs.append(self._frame_seq_next)
        self._frame_seq_next += 1
        self._frame_stats["recorded"] += 1
        if wrist is not None:
            wrist_jpeg = encode_frame(wrist)
            self._wrist_frame_buffer.append(wrist_jpeg)
            self._buffer_bytes += len(wrist_jpeg)
        if self._frame_listener is not None:
            self._frame_listener(frame)
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
        """フレーム（メイン + 手首）の JPEG の合計バイト数の上限。None で無制限。"""
        self._frame_budget_bytes = budget_bytes

    def get_frame_stats(self) -> dict:
        """録画の記録。録った枚数・捨てた枚数・残った枚数・間隔の変更履歴。"""
        retained = len(self._frame_buffer)
        return {
            "recorded": self._frame_stats["recorded"],
            "dropped": self._frame_stats["dropped"],
            "retained": retained,
            "buffered_bytes": self._buffer_bytes,
            "subsample_rate": self._subsample_rate,
            "rate_changes": list(self._frame_stats["rate_changes"]),
        }

    def _clear_frames(self) -> None:
        self._frame_buffer.clear()
        self._wrist_frame_buffer.clear()
        self._buffer_bytes = 0
        self._frame_seqs.clear()
        self._frame_seq_next = 0
        self._pinned_seqs.clear()

    def _sync_seqs(self) -> None:
        """通し番号とメインのバッファの数がずれていたら、合わせ直す。

        メインだけを取り出した直後（手首がまだ残っている）は、ずれていて正しい。
        """
        if len(self._frame_seqs) == len(self._frame_buffer):
            return
        if not self._frame_buffer:
            if self._wrist_frame_buffer:
                return
            self._clear_frames()
        else:
            self._frame_seqs = list(range(len(self._frame_buffer)))
            self._frame_seq_next = len(self._frame_buffer)

    def _jpegs_in(self, buffer: list[bytes], start: int, end: int) -> list[bytes]:
        self._sync_seqs()
        if len(buffer) != len(self._frame_seqs):  # 手首カメラを録っていない
            return []
        return [f for seq, f in zip(self._frame_seqs, buffer) if start <= seq < end]

    def get_video_frames(self, *, clear: bool = False) -> list[np.ndarray]:
        """メインのフレームを、画像の配列に戻して返す。

        `clear=True` はメインだけを空にする。手首のフレームは
        `get_wrist_video_frames(clear=True)` で別に取り出す。
        """
        frames = [decode_frame(f) for f in self._frame_buffer]
        if clear:
            self._buffer_bytes -= sum(len(f) for f in self._frame_buffer)
            self._frame_buffer = []
            if not self._wrist_frame_buffer:
                self._frame_seqs.clear()
                self._pinned_seqs.clear()
                self._frame_seq_next = 0
        return frames

    def get_video_frame_count(self) -> int:
        """次に録るフレームの通し番号。ターンの境界のフレームを間引きから守る。"""
        self._sync_seqs()
        n = self._frame_seq_next
        # そのターンの最初のフレーム（n）と、前のターンの最後のフレーム（n-1）
        self._pinned_seqs.update((n - 1, n))
        return n

    def get_video_frames_range(self, start: int, end: int) -> list[np.ndarray]:
        return [decode_frame(f) for f in self._jpegs_in(self._frame_buffer, start, end)]

    def iter_video_frames_range(self, start: int, end: int) -> Iterator[np.ndarray]:
        """`get_video_frames_range` と同じものを、1 枚ずつ戻して返す。

        戻した画像は 1 枚あたり生の大きさになる。まとめて戻すと、JPEG で節約した
        メモリを取り出しの時点で使ってしまう。動画に書くときはこちらを使う。
        """
        for data in self._jpegs_in(self._frame_buffer, start, end):
            yield decode_frame(data)

    def get_wrist_video_frames(self, *, clear: bool = False) -> list[np.ndarray]:
        frames = [decode_frame(f) for f in self._wrist_frame_buffer]
        if clear:
            self._buffer_bytes -= sum(len(f) for f in self._wrist_frame_buffer)
            self._wrist_frame_buffer = []
            if not self._frame_buffer:
                self._frame_seqs.clear()
                self._pinned_seqs.clear()
                self._frame_seq_next = 0
        return frames

    def get_wrist_video_frames_range(self, start: int, end: int) -> list[np.ndarray]:
        return [decode_frame(f) for f in self._jpegs_in(self._wrist_frame_buffer, start, end)]

    def _buffered_bytes(self) -> int:
        return self._buffer_bytes

    def _thin_frames(self) -> None:
        """上限を超えたら、1 つ飛ばしに捨てて、以後の録画間隔を倍にする。

        最新のフレーム・最初のフレーム・ターンの境界のフレームは残す。
        残すべきものだけで上限を超える場合は、それ以上は捨てない。
        """
        budget = self._frame_budget_bytes
        if budget is None:
            return
        while self._buffer_bytes > budget and len(self._frame_buffer) > 2:
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
            self._buffer_bytes = sum(len(f) for f in self._frame_buffer) + sum(
                len(f) for f in self._wrist_frame_buffer
            )
            self._subsample_rate *= 2
            self._frame_stats["rate_changes"].append(
                {"at_seq": self._frame_seq_next, "subsample_rate": self._subsample_rate}
            )
