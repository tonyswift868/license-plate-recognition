# 派单 02 · 员工 B（WP-B）· 第二批：仿 CCPD 数据集与版式校准脚本

```text
你是“停车场收费系统”M2 车牌识别模块的执行者【员工 B / WP-B，工位 02】。
工作目录：D:\keshe2\workers\02    分支：feature/m2-wpb-locate（已检出，不要切换分支）
环境：工作目录自带 .venv（Python 3.12、OpenCV 4.14，依赖已装好，lpr 已可编辑安装）。所有命令在工作目录根下运行，
      用 .venv\Scripts\python.exe。
必读：docs/M2-车牌识别-开发文档.md 第 2、5 章，6.3—6.4 节，7.1—7.5 节，11.3 节，12.3 节，第 15 章，附录 A、B；
      algorithm/lpr_tools/data/ccpd.py（员工 C 已实现的 CCPD 文件名解析与索引，你的输出必须能被它正确解析）；
      algorithm/lpr_tools/synth_scenes.py（你上一批的合成场景生成器）
背景：CCPD 还没到位。本批目标是“数据一到就能直接跑”：造一份目录结构、文件名格式都与 CCPD 一致的仿真数据集，
      并写好版式校准脚本，用仿真数据跑通。仿真数据上的数字只证明流程正确，不作为正式指标。

本次任务（按顺序做，每做完一项提交一次）：
  B-5 仿 CCPD 数据集生成
      - 给 lpr_tools/synth_scenes.py 增加 --ccpd 模式（或新建 lpr_tools/synth_ccpd.py），输出到 algorithm/data/raw_mock/：
        · ccpd2019/ccpd_base、ccpd_db（偏暗）、ccpd_tilt（大角度透视）、ccpd_blur（模糊）、ccpd_np（无车牌）
          以及 ccpd2019/splits/train.txt、val.txt、test.txt、ccpd_db.txt 等划分文件；
        · ccpd2020/ccpd_green/{train,val,test}（8 位绿牌）。
      - 图像为竖幅 720×1160（与 CCPD 一致）；文件名严格按附录 B 的 7 段格式编码：面积、倾斜、bbox、
        四角点（RB_LB_LT_RT 顺序）、车牌下标（用 CCPD 官方解码表，派单 03 首批里有完整表）、亮度、模糊度。
      - 省份汉字允许用 Pillow + C:\Windows\Fonts\simhei.ttf 渲染（架构师已批准：Pillow 可用于 lpr_tools 下的
        开发期合成；运行时包 lpr 仍不得引用 Pillow）；车牌内容随机但必须符合附录 A 的版式。
      - 规模参数化（如 --per-subset 50 --seed 42），默认总量控制在 500 张以内。
      - 单测：生成少量样本到 tmp_path，用 lpr_tools.data.ccpd.parse_ccpd_name / build_index 解析，
        车牌号、quad、划分都必须与生成时的真值一致。
  B-6 版式校准脚本（任务 B3 的数据部分，数据到位后直接用）
      - lpr_tools/calib_layout.py：读 ccpd_index.csv（lpr_tools.data.ccpd.read_index）→ 用真值 quad 透视校正到
        252×80 / 276×80 → charnorm.unify_polarity → 二值化 → 竖直投影统计每一位字符的中心与宽度（相对板宽）→
        输出建议的 layout7_centers / layout7_char_w / layout8_centers / layout8_char_w、各位的标准差，
        以及“真值定位下分割位数正确率”（用当前 segment()）。只打印和写报告文件，不自动改 default.toml。
      - 命令：python -m lpr_tools.calib_layout --index <ccpd_index.csv> --raw <raw 根目录> --n 2000 --out algorithm/out/calib
      - 在仿 CCPD 数据上跑通；单测用小样本。
  B-7（时间允许再做）倾斜残余校正：在 rectify 或 segment 中加入垂直错切搜索（6.3：[-0.3, 0.3] 内找竖直投影谷最深的系数），
      默认关闭或以不降低现有测试结果为前提；在 ccpd_tilt 仿真子集上对比开 / 关的分割位数正确率。

只允许新建或修改：
  algorithm/lpr/preprocess.py、locate.py、rectify.py、segment.py
  algorithm/lpr_tools/synth_scenes.py、algorithm/lpr_tools/synth_ccpd.py、algorithm/lpr_tools/calib_layout.py、algorithm/lpr_tools/viz.py
  algorithm/config/default.toml —— 只改 [preprocess] [locate] [rectify] [segment] 段的值（不改键名、不改冻结值）
  algorithm/tests/ 下以 test_preprocess、test_locate、test_rectify、test_segment、test_viz、test_synth、test_calib 开头的文件
禁止：
  - 修改 types.py、charset.py、charnorm.py、errors.py、config.py、pipeline.py、testing.py、cli.py、recognize/ 下任何文件、
    lpr_tools/data/ 下任何文件、tests/conftest.py、tests/test_contracts.py、tests/test_pipeline.py
  - 删除或重命名配置键；引入白名单外的依赖，不得 pip install 任何东西；使用任何车牌识别 API 或预训练模型
  - 对可能含中文的路径使用 cv2.imread / cv2.imwrite（一律用 lpr.io_utils）
  - 提交 algorithm/data/、algorithm/out/ 下的任何文件
  - 切换或合并分支，改动 D:\keshe2 主检出或其他 workers 目录
协作方式：
  积极派子代理并行完成相互独立的子任务（如不同文件的实现、单测、验收命令），以加快进度；
  子代理同样只能改本派单允许的文件，两个子代理不得同时改同一个文件；汇总、自测与提交由你本人负责。
  需要看图时先缩小到宽 ≤800 px 再打开，一次最多看几张；不要在对话里打开大图或拼图（会拖垮对话界面）。
提交：只在本分支提交，提交信息用 feat(m2): / test(m2): / fix(m2): 前缀。

验收命令（必须实际运行，原始输出保存到 algorithm/out/receipts/ 下）：
  .venv\Scripts\python.exe -m pytest algorithm/tests -q
  .venv\Scripts\python.exe -m lpr_tools.synth_scenes --ccpd --out algorithm/data/raw_mock --per-subset 50 --seed 42
      （若新建了 synth_ccpd.py，用对应命令）
  .venv\Scripts\python.exe -m lpr_tools.data.ccpd --raw algorithm/data/raw_mock --out algorithm/data/index_mock/ccpd_index.csv
  .venv\Scripts\python.exe -m lpr_tools.calib_layout --index algorithm/data/index_mock/ccpd_index.csv --raw algorithm/data/raw_mock --n 400 --out algorithm/out/calib
  .venv\Scripts\python.exe -m lpr.cli algorithm/data/raw_mock/ccpd2019/ccpd_base --uniform --json

汇报方式：
  完整汇报写到 algorithm/out/receipts/第二批.md（含：变更文件列表与最后 commit 号、实现要点≤10 行、参数及来源、
  验收命令原始输出或输出文件路径、已知问题与未完成项、需要架构师决策的事项）。
  聊天里只回一行：回执 algorithm/out/receipts/第二批.md，commit <最后提交号>
```
