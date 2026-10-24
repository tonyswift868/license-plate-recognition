"""M2 车牌识别算法包 lpr（见开发文档第 5 章）。

    from lpr import PlateRecognizer, RecognitionResult, RecogStatus, DuplicateFilter, PlateVoteSession
    from lpr.testing import FakeRecognizer
"""

from ._version import __version__
from .dedup import DuplicateFilter
from .errors import ConfigError, ModelLoadError
from .pipeline import PlateRecognizer
from .types import RecognitionResult, RecogStatus
from .voting import PlateVoteSession

__all__ = [
    "__version__",
    "ConfigError",
    "DuplicateFilter",
    "ModelLoadError",
    "PlateRecognizer",
    "PlateVoteSession",
    "RecogStatus",
    "RecognitionResult",
]
