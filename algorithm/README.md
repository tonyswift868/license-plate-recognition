# algorithm/ — M2 车牌识别

纯 Python 算法包 `lpr`：输入车辆图像，输出车牌号、置信度、校正图和状态码。设计与分工见 `docs/M2-车牌识别-开发文档.md`。

## 快速开始（Windows PowerShell，在仓库根目录）

```powershell
uv venv --seed --python 3.12 .venv            # 或 python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r algorithm/requirements.txt
pip install -e ./algorithm
pytest algorithm/tests -q
```

每个 worktree（`workers/01`、`02`、`03`）都有自己的 `.venv`，在哪个目录工作就激活哪个目录的环境。
`pytest` 已配置为优先导入当前检出的 `algorithm/`，不会误用别的目录里的代码。

## 命令速查

| 用途 | 命令 |
| --- | --- |
| 全量测试 | `pytest algorithm/tests -q` |
| 契约测试 | `pytest algorithm/tests/test_contracts.py -q` |
| 覆盖率（G7） | `pytest algorithm/tests --cov=lpr --cov-report=term-missing --cov-fail-under=60` |
| 识别单图 / 目录 | `python -m lpr.cli 图片或目录 [--lane 1] [--json]` |
| 用联调桩识别 | `python -m lpr.cli 图片或目录 --fake` |
| 真实流水线、无模型 | `python -m lpr.cli 图片或目录 --uniform --debug-dir algorithm/out/dbg` |
| 多帧（同一车辆） | `python -m lpr.cli 图1 图2 图3 --many` |
| 导出中间图 | `python -m lpr.cli 图片 --debug-dir algorithm/out/dbg` |
| 切换分类器 | `python -m lpr.cli 图片 --classifier svm --model models/svm_v1.joblib` |
| G1 单帧耗时 | `python -m lpr_tools.eval.bench_latency --manifest algorithm/testset/manifest.csv --n 100 --size 1280x720` |
| G2 准确率 | `python -m lpr_tools.eval.eval_accuracy --manifest algorithm/testset/manifest.csv --split val` |
| G3 视频吞吐 | `python -m lpr_tools.eval.bench_video --video <视频> --sample-fps 5` |

评测与基准脚本由员工 C 提供，就绪前上表最后三行不可用。

## 在后端里调用（M3）

```python
from lpr import PlateRecognizer
from lpr.testing import FakeRecognizer

recognizer = FakeRecognizer() if settings.LPR_FAKE else PlateRecognizer.from_config()
recognizer.warmup()
result = await asyncio.to_thread(recognizer.recognize_bytes, data, lane_id=lane_id)
payload = result.to_dict()
```

- `.env`：`LPR_CONFIG`（配置文件路径）、`LPR_FAKE`（1 = 使用桩）。
- `recognize*` 永不抛异常，只看 `result.status`；只有 `from_config` 会抛 `ModelLoadError` / `ConfigError`。
- 多张图（同一车辆）调 `recognize_many`，元素可以是字节或 BGR 数组。

## 目录

| 路径 | 内容 | 归属 |
| --- | --- | --- |
| `lpr/` | 运行时包（后端唯一依赖入口） | 见开发文档 9.1 |
| `lpr_tools/` | 数据、训练、评测工具，不进后端运行时 | 见开发文档 9.1 |
| `config/default.toml` | 全部算法参数（键名冻结） | 架构师 |
| `models/` | 模型权重与 `manifest.json` | 员工 A |
| `tests/` | 契约测试与各模块单测 | 架构师 / 各执行者 |
| `reports/scoreboard.md` | 每日积分板 | 架构师 |
| `data/` | 数据集（不入库） | 人工 |

## 模型登记

训练脚本保存模型后必须调用 `lpr.recognize.write_manifest_entry()` 登记到 `models/manifest.json`。
加载时会校验字符表哈希和文件 sha256，对不上就拒绝加载。
