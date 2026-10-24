"""重复抓拍判定 E7（员工 C，任务 C5；规则见开发文档 6.8）。"""

from __future__ import annotations

import threading

from .config import LprConfig
from .types import RecognitionResult


class DuplicateFilter:
    """M2 只给出判定；是否丢弃、停车会话如何幂等由 M3 决定。有状态，内部加锁。"""

    def __init__(self, window_s: float = 30, dhash_max_hamming: int = 10) -> None:
        self.window_s = window_s
        self.dhash_max_hamming = dhash_max_hamming
        self._lock = threading.Lock()

    @classmethod
    def from_config(cls, cfg: LprConfig) -> "DuplicateFilter":
        return cls(cfg.dedup.window_s, cfg.dedup.dhash_max_hamming)

    def check(self, lane_id: int, result: RecognitionResult, ts: float) -> tuple[bool, str]:
        """返回 (是否重复, 原因)。原因为 "same_plate"、"similar_plate"，不重复时为 ""。

        不重复时记录本次结果，供后续判定。ts 为秒。
        """
        raise NotImplementedError("DuplicateFilter 尚未实现（员工 C，任务 C5）")

    def reset(self) -> None:
        raise NotImplementedError("DuplicateFilter 尚未实现（员工 C，任务 C5）")
