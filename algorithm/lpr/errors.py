"""状态与 3xxx 错误码、异常类型（冻结，见开发文档 5.3）。

接口文档 docs/04-接口文档.md 发布细分码后（17.2 W1），只调整 ERROR_CODES 的编号映射，状态名不变。
"""

from __future__ import annotations

from .types import RecogStatus, RecognitionResult

ERROR_CODES: dict[RecogStatus, int] = {
    RecogStatus.OK: 0,
    RecogStatus.NO_PLATE: 3001,
    RecogStatus.LOW_CONFIDENCE: 3002,
    RecogStatus.SEG_FAILED: 3003,
    RecogStatus.INVALID_FORMAT: 3004,
    RecogStatus.BAD_INPUT: 3005,
    RecogStatus.UNSUPPORTED: 3006,
    RecogStatus.INTERNAL_ERROR: 3099,
}

DEFAULT_MESSAGES: dict[RecogStatus, str] = {
    RecogStatus.OK: "识别成功",
    RecogStatus.NO_PLATE: "未检出车牌，请人工录入",
    RecogStatus.LOW_CONFIDENCE: "识别置信度低，请人工确认",
    RecogStatus.SEG_FAILED: "车牌字符分割失败，请人工录入",
    RecogStatus.INVALID_FORMAT: "识别结果不符合车牌版式，请人工确认",
    RecogStatus.BAD_INPUT: "图像无效，请重新抓拍",
    RecogStatus.UNSUPPORTED: "特种车牌暂不支持，请人工录入",
    RecogStatus.INTERNAL_ERROR: "算法内部异常，请人工录入",
}


def error_code(status: RecogStatus) -> int:
    return ERROR_CODES[status]


class LprError(Exception):
    """lpr 包异常基类。"""


class ModelLoadError(LprError):
    """模型缺失、manifest 不匹配或分类器无法加载。只会从 PlateRecognizer.from_config 抛出。"""


class ConfigError(LprError):
    """配置文件缺失、键名错误或取值非法。只会从 PlateRecognizer.from_config 抛出。"""


class BadInputError(LprError, ValueError):
    """阶段一发现输入无效（短边过小等）时抛出，流水线转为 BAD_INPUT，不会传到调用方。"""


def failure_result(status: RecogStatus, message: str | None = None, *,
                   elapsed_ms: float = 0.0, stage_ms: dict[str, float] | None = None,
                   frames_used: int = 1, model_version: str = "") -> RecognitionResult:
    """构造没有车牌号的结果（NO_PLATE、SEG_FAILED、UNSUPPORTED、BAD_INPUT、INTERNAL_ERROR）。"""
    return RecognitionResult(
        status=status,
        error_code=ERROR_CODES[status],
        message=message or DEFAULT_MESSAGES[status],
        plate_no=None,
        plate_color="unknown",
        layout=None,
        confidence=0.0,
        char_confidences=[],
        candidates=[],
        bbox=None,
        corners=None,
        plate_image=None,
        elapsed_ms=elapsed_ms,
        stage_ms=dict(stage_ms or {}),
        frames_used=frames_used,
        model_version=model_version,
    )
