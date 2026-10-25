"""三种字符分类器共用的协议（冻结，见开发文档 5.2）。"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from ..types import PlateCandidate


@runtime_checkable
class CharClassifier(Protocol):
    """字符分类器。实现：recognize/cnn.py、template.py、svm.py（员工 A）。

    约定：
    - 每个实现提供 classmethod `from_file(path: Path, *, version: str)`，由 recognize/__init__.py 调用；
    - predict_proba 必须线程安全、不修改共享状态（5.6）；PyTorch 实现须处于 eval 模式，
      推理包在 torch.inference_mode() 中；
    - 输入为空（0×32×32）时返回 0×65。
    """

    name: str
    version: str

    def predict_proba(self, chars: np.ndarray) -> np.ndarray:
        """输入 N×32×32 uint8，输出 N×65 float32，每行和为 1，列顺序即 CHARSET 下标。"""
        ...


@runtime_checkable
class CornerDetector(Protocol):
    """四角点网络（开发文档 14 章 K1 Plan B-1，员工 B）。实现：lpr/corner_net.py 的 CornerNet。

    约定同 CharClassifier：提供 classmethod `from_file(path: Path, *, version: str)`，线程安全，eval + inference_mode。
    """

    name: str
    version: str

    def detect(self, image: np.ndarray, max_candidates: int) -> list[PlateCandidate]:
        """原图 BGR uint8 → 按 score 降序的车牌候选（原图坐标，corners 为 LT,RT,RB,LB，source="net"）。
        只返回有把握的候选（通常 0—2 个）；流水线把它们排在传统定位候选之前。"""
        ...


@runtime_checkable
class PlateReader(Protocol):
    """整牌识别 CRNN+CTC（开发文档 14 章 K1 Plan B-2，员工 A）。实现：lpr/recognize/crnn.py 的 CrnnReader。

    约定同 CharClassifier：提供 classmethod `from_file(path: Path, *, version: str)`，线程安全，eval + inference_mode。
    """

    name: str
    version: str

    def read(self, plate_image: np.ndarray) -> tuple[str, list[float]]:
        """校正后的整块车牌 BGR（7 位 252×80、8 位 276×80）→ (车牌字符串, 逐位置信度)。
        字符只能取自 CHARSET；读不出时返回 ("", [])。状态判定由流水线按 5.4 与附录 A 统一完成。"""
        ...
