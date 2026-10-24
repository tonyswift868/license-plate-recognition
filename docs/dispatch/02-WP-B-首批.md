# 派单 02 · 员工 B（WP-B 定位·校正·分割）· 首批

```text
你是“停车场收费系统”M2 车牌识别模块的执行者【员工 B / WP-B，工位 02】。
工作目录：D:\keshe2\workers\02    分支：feature/m2-wpb-locate（已检出，不要切换分支）
环境：工作目录自带 .venv（Python 3.12，依赖已装好，lpr 已可编辑安装）。所有命令在工作目录根下运行，
      用 .venv\Scripts\python.exe，例如 .venv\Scripts\python.exe -m pytest algorithm/tests -q
必读：docs/M2-车牌识别-开发文档.md 第 2、4、5 章（含 5.9），6.1—6.4 节，11.3 节，第 15 章，附录 A；
      algorithm/lpr/types.py、charnorm.py、config.py、pipeline.py（看各阶段怎么被调用）；algorithm/config/default.toml
背景：CCPD 数据集还没到位。本批任务全部用合成图完成；不要等数据，也不要自己下载任何数据集。
      B1 的召回指标、B3 的版式校准等正式验收等 CCPD 到位后再做。

本次任务（按顺序做，每做完一项提交一次）：
  B-0 合成场景生成器
      - lpr_tools/synth_scenes.py：生成带车牌的合成整车场景图（1280×720 BGR），覆盖蓝/黄/绿牌、7 位/8 位、
        透视倾斜（±25°）、尺度变化、暗光、噪声、背景干扰（含蓝色车身）；同时输出真值 labels.csv
        （file,plate_text,color,layout,quad；quad 为原图坐标 LT,RT,RB,LB）。字符用 cv2 画字母数字即可（cv2 不能写中文，
        首位可用一个方块代替省份字）。
        命令：python -m lpr_tools.synth_scenes --out algorithm/out/synth --n 40 --seed 42
      - 你的单测可以直接 import 它生成小批量样本。
  B-1 预处理（完整）+ 定位 v0（任务 B1 的代码部分）
      - preprocess.py：严格按 6.1；单通道/H×W×1 转 3 通道；短边 < min_side 抛 errors.BadInputError；
        原图更窄时不放大；HSV 用未做 CLAHE 的图；正确记录 scale 与 offset（车道 ROI）。
      - locate.py：严格按 6.2（颜色三分支 + Sobel-x 边缘分支、形态学、minAreaRect、几何过滤、打分、NMS、
        坐标换算回原图、corners 顺序 LT,RT,RB,LB、候选颜色多数投票）。
  B-2 校正 v0 + 颜色判别（任务 B2 的代码部分）
      - rectify.py：严格按 6.3；在原图上取 expand 10% 的区域精修四点，失败退回 minAreaRect；
        7 位 252×80、8 位 276×80；homography 为原图 → 校正图。
  B-3 分割 v0（任务 B3 的代码部分）
      - segment.py：严格按 6.4；极性统一必须用 charnorm.unify_polarity，字符图必须用 charnorm.normalize_char；
        失败返回 ok=False 与 0×32×32，不抛异常。
      - 7 位用 default.toml 里的初值。8 位版式参数：按公开的 GA 36-2018 新能源号牌尺寸推算临时值，
        写入 default.toml 的 segment.layout8_centers（8 个）与 layout8_char_w，并在注释里标【临时值，待 CCPD2020 校准】。
  B-4 可视化（任务 B5）
      - lpr_tools/viz.py：对一张图输出各阶段中间图（颜色掩膜、边缘掩膜、候选框、校正图、二值图、分割框），
        调试图上不写中文（用下标或拼音）；图片读写一律用 lpr.io_utils（中文路径安全）。
        命令：python -m lpr_tools.viz <图片...> --out algorithm/out/viz
  每项都要有单测：tests/test_preprocess.py、test_locate.py、test_rectify.py、test_segment.py、test_viz.py，
  每个阶段至少覆盖正常样例、空或异常输入、边界情况（极小车牌、全黑图）。

只允许新建或修改：
  algorithm/lpr/preprocess.py、locate.py、rectify.py、segment.py
  algorithm/lpr_tools/viz.py、algorithm/lpr_tools/synth_scenes.py
  algorithm/config/default.toml —— 只改 [preprocess] [locate] [rectify] [segment] 段的值（不改键名；
    rectify.size_7、size_8、segment.char_size 是冻结值，不能改）
  algorithm/tests/test_preprocess.py、test_locate.py、test_rectify.py、test_segment.py、test_viz.py
  algorithm/notebooks/ 下的文件
禁止：
  - 修改 types.py、charset.py、charnorm.py、errors.py、config.py、pipeline.py、testing.py、cli.py、
    recognize/ 下任何文件、tests/conftest.py、tests/test_contracts.py、tests/test_pipeline.py
  - 删除或重命名配置键（需要新键时在汇报里申请）
  - 引入白名单外的依赖（见 algorithm/requirements.txt，不允许 scikit-image），不得 pip install 任何东西；
    调用任何车牌识别 API 或使用预训练车牌/检测模型（YOLO/ultralytics、HyperLPR 等）
  - 对可能含中文的路径使用 cv2.imread / cv2.imwrite（一律用 lpr.io_utils）
  - 提交 algorithm/data/、algorithm/out/ 下的任何文件
  - 切换或合并分支，改动 D:\keshe2 主检出或其他 workers 目录
提交：只在本分支提交，提交信息用 feat(m2): / test(m2): / fix(m2): 前缀。

验收命令（必须实际运行，并贴出完整原始输出）：
  .venv\Scripts\python.exe -m pytest algorithm/tests -q
  .venv\Scripts\python.exe -m lpr_tools.synth_scenes --out algorithm/out/synth --n 40 --seed 42
  .venv\Scripts\python.exe -m lpr.cli algorithm/out/synth --uniform --json
  .venv\Scripts\python.exe -m lpr_tools.viz algorithm/out/synth --out algorithm/out/viz
  另外写一段一次性脚本（不入库）统计：40 张合成图上 Top-3 定位召回（任一候选与真值 quad 外接框 IoU ≥0.5）、
  分割位数正确率，以及各阶段耗时 P95（与 4.2 预算对比），把脚本内容和输出一起贴出。
  说明：本分支里阶段六 decode 还是桩，lpr.cli --uniform 的结果会是 INTERNAL_ERROR（“阶段六 decode 尚未实现”），
  这是正常的；看 stage_ms 里 preprocess—recognize 都有耗时，就说明阶段一到四已跑通。

汇报格式：
  1. 变更文件列表（以及最后一个 commit 号）
  2. 实现要点（不超过 10 行）
  3. 参数初值及来源（含 8 位版式临时值的推算过程）
  4. 验收命令原始输出
  5. 已知问题与未完成项
  6. 需要架构师决策的事项
```
