# 派单 03 · 员工 C（WP-C）· 第二批：测试集抽样、每日评测与金样例脚本

```text
你是“停车场收费系统”M2 车牌识别模块的执行者【员工 C / WP-C，工位 03】。
工作目录：D:\keshe2\workers\03    分支：feature/m2-wpc-post-eval（已检出，不要切换分支）
环境：工作目录自带 .venv（Python 3.12、OpenCV 4.14，依赖已装好，lpr 已可编辑安装）。所有命令在工作目录根下运行，
      用 .venv\Scripts\python.exe。
必读：docs/M2-车牌识别-开发文档.md 第 2、5 章，7.4 节，10.2 节，11.4 节，12.1 节（金样例构成），12.3 节，第 13 章，附录 B；
      你上一批的 lpr_tools/data/ccpd.py、manifest.py、lpr_tools/eval/
背景：CCPD 还没到位。本批目标是“数据一到就能直接跑”：测试集 v1 抽样、每日验证集评测、定位召回指标、金样例挑选
      全部写成脚本，用自己构造的小样本跑通。员工 B 同时在造仿 CCPD 数据集，你不依赖它，单测用自己构造的样本。

本次任务（按顺序做，每做完一项提交一次）：
  C-6 测试集 v1 抽样脚本（任务 C1 的数据部分）
      - lpr_tools/data/sample_testset.py，严格按 7.4：从 ccpd_index.csv 抽样生成 testset/manifest.csv：
        清晰 = ccpd_base 官方 test 划分 120 + CCPD2020 test 30；夜间 = ccpd_db 中亮度低于阈值的 50（阈值作为参数，
        默认值写在脚本里并在汇报中说明依据）；倾斜 = ccpd_tilt + ccpd_rotate 50；污损 = ccpd_blur / weather / challenge
        候选 50（note 写“待人工确认”）；无牌 = ccpd_np 20；本地 = 读 raw/local/labels.csv（若存在，全收）。
        固定种子 42；绝不抽 train / val 划分的图；输出后调用 manifest.py 校验；打印各子类数量。
      - 命令：python -m lpr_tools.data.sample_testset --index <ccpd_index.csv> --out algorithm/testset/manifest.csv
        本批只在 tmp 或 algorithm/out/ 下生成演示清单，不提交正式 manifest.csv（等真实数据）。
  C-7 每日评测补全（任务 C2 v1）
      - eval_accuracy 的 --split val：从 ccpd_index 的 val 划分按子集分层抽样（--n、--seed 42），作为每日验证集；
      - 新增 --locate-recall：统计 Top-3 定位召回（任一候选 bbox 与索引真值 bbox 的 IoU ≥ 0.5），按子集输出，
        这是任务 B1 的验收指标；
      - --scoreboard 追加一行时自动填日期、git commit 短号、分类器（本批仍只写临时副本，不改正式积分板）。
  C-8 金样例挑选脚本（替架构师准备，12.1 第 5 条）
      - lpr_tools/data/pick_golden.py：从 ccpd_index 按 10 base、2 green、2 db、2 tilt、2 blur、2 np 挑 20 张
        （只从 test 划分挑，种子 42），复制到指定目录并写 labels.csv（列 file,plate_no,plate_color,category；
        np 行 plate_no 为空）。本批只在 tmp / algorithm/out/ 下演示，不往 tests/data/golden/ 放图。

只允许新建或修改：
  algorithm/lpr/plate_rules.py、postprocess.py、voting.py、dedup.py
  algorithm/lpr_tools/data/ccpd.py、manifest.py、sample_testset.py、pick_golden.py
  algorithm/lpr_tools/eval/ 下的文件（ablation.py 除外）
  algorithm/tests/ 下以 test_plate_rules、test_postprocess、test_voting、test_dedup、test_ccpd、test_manifest、
    test_eval、test_bench、test_sample、test_golden_pick 开头的文件
禁止：
  - 修改 types.py、charset.py、charnorm.py、errors.py、config.py、pipeline.py、testing.py、cli.py、recognize/ 下任何文件、
    tests/conftest.py、tests/test_contracts.py、tests/test_pipeline.py、tests/test_golden.py、config/default.toml
  - 修改 algorithm/testset/manifest.csv、algorithm/reports/scoreboard.md、tests/data/golden/
  - 引入白名单外的依赖（不允许 pandas），不得 pip install 任何东西；使用任何车牌识别 API 或预训练模型
  - 提交 algorithm/data/、algorithm/out/ 下的任何文件
  - 切换或合并分支，改动 D:\keshe2 主检出或其他 workers 目录
协作方式：
  积极派子代理并行完成相互独立的子任务（如不同文件的实现、单测、验收命令），以加快进度；
  子代理同样只能改本派单允许的文件，两个子代理不得同时改同一个文件；汇总、自测与提交由你本人负责。
  需要看图时先缩小到宽 ≤800 px 再打开，一次最多看几张；不要在对话里打开大图或拼图（会拖垮对话界面）。
提交：只在本分支提交，提交信息用 feat(m2): / test(m2): / fix(m2): 前缀。

验收命令（必须实际运行，原始输出保存到 algorithm/out/receipts/ 下）：
  .venv\Scripts\python.exe -m pytest algorithm/tests -q
  .venv\Scripts\python.exe -m pytest algorithm/tests --cov=lpr --cov-report=term
  .venv\Scripts\python.exe -m lpr_tools.data.sample_testset --help
  .venv\Scripts\python.exe -m lpr_tools.data.pick_golden --help
  .venv\Scripts\python.exe -m lpr_tools.eval.eval_accuracy --help
  另附：三个脚本在你构造的小索引上的完整运行输出（含 --split val、--locate-recall 用 --uniform 跑的结果）。

汇报方式：
  完整汇报写到 algorithm/out/receipts/第二批.md（含：变更文件列表与最后 commit 号、实现要点≤10 行、参数及来源、
  验收命令原始输出或输出文件路径、已知问题与未完成项、需要架构师决策的事项）。
  聊天里只回一行：回执 algorithm/out/receipts/第二批.md，commit <最后提交号>
```
