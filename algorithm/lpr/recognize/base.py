"""三种字符分类器共用的协议（冻结，见开发文档 5.2）。"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


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
