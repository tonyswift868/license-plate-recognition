"""多帧投票与流式投票会话（员工 C，任务 C5；算法见开发文档 6.7）。"""

from __future__ import annotations

import threading
from collections.abc import Sequence

from .config import LprConfig
from .types import RecognitionResult


def vote(results: Sequence[RecognitionResult], *, min_frames: int = 2,
         min_confidence: float = 0.80) -> RecognitionResult:
    """同一车辆多帧结果逐位加权投票。

    有效帧（OK、LOW_CONFIDENCE）少于 min_frames 时返回置信度最高的单帧结果；
    全部无效时返回第一帧结果。返回值的 frames_used 为参与投票的帧数，status 按 6.6 规则重新判定。
    """
    raise NotImplementedError("vote 尚未实现（员工 C，任务 C5）")


class PlateVoteSession:
    """流式投票：逐帧 feed；同一车牌 Top-1 累计 agree_frames 帧，或距首帧超过 window_s 秒，就输出一次结果并重置。

    有状态，内部加锁；每个车道各建一个实例（5.6）。
    """

    def __init__(self, agree_frames: int = 3, window_s: float = 2.0, *,
                 min_frames: int = 2, min_confidence: float = 0.80) -> None:
        self.agree_frames = agree_frames
        self.window_s = window_s
        self.min_frames = min_frames
        self.min_confidence = min_confidence
        self._lock = threading.Lock()

    @classmethod
    def from_config(cls, cfg: LprConfig) -> "PlateVoteSession":
        return cls(cfg.voting.agree_frames, cfg.voting.window_s,
                   min_frames=cfg.voting.min_frames, min_confidence=cfg.postprocess.min_confidence)

    def feed(self, result: RecognitionResult, ts: float) -> RecognitionResult | None:
        """喂入一帧（ts 为秒）；达到输出条件时返回投票结果，否则返回 None。"""
        raise NotImplementedError("PlateVoteSession 尚未实现（员工 C，任务 C5）")

    def flush(self) -> RecognitionResult | None:
        """立即输出当前累积帧的投票结果并重置；没有累积帧时返回 None。"""
        raise NotImplementedError("PlateVoteSession 尚未实现（员工 C，任务 C5）")

    def reset(self) -> None:
        raise NotImplementedError("PlateVoteSession 尚未实现（员工 C，任务 C5）")
