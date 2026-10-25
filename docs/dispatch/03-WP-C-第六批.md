# 派单 03 · 员工 C · 第六批：每日评测一键脚本、E1 阈值、结果仲裁分析

```text
你是 M2 车牌识别模块执行者【员工 C，工位 03】。工作目录 D:\keshe2\workers\03，分支 feature/m2-wpc-post-eval，用 .venv\Scripts\python.exe。
必读：algorithm/reports/scoreboard.md 前两行；docs/M2-车牌识别-开发文档.md 5.4 节（置信度与 E1 阈值）、10.2 节（日常节奏）、13 章；algorithm/lpr/pipeline.py（结果怎么挑）。
现状（develop，cnn v1）：val 整牌 44.6%，min_confidence=0.80 时自动通过率 29.2%、自动通过精度 100%。
任务：
  1. 每日评测一键脚本 lpr_tools/eval/daily.py：一条命令依次跑 val 1000、难例开发集 7×300、金样例、G1（测试清单 1280×720×100），
     输出一行可直接贴进积分板的 markdown（列与 scoreboard.md 表头一致）、逐位置错误与混淆 Top20、各子集表，并与上一次运行对比，
     任何类别退步 >1 个百分点要醒目标出。结果写到 algorithm/out/daily/<日期-commit>/，不改 scoreboard.md。
  2. E1 阈值（5.4）：在 val 上（cnn v1）扫描 min_confidence，画“自动通过率—自动通过精度”曲线，找出精度 ≥99% 前提下自动通过率最高的阈值，
     给出建议值和依据（架构师决定是否采用）。只用 val，绝不用测试清单。
  3. 结果仲裁分析：复现流水线对每张 val 图的全部尝试（每个候选、每个版式、每个四角点假设的解码结果），统计“最终挑错了、但别的尝试里有正确车牌”的张数，
     按原因分类（例：错误的绿/未知色候选的 8 位结果压过正确蓝牌 7 位结果），并提出排序规则改进建议与预计收益（流水线编排由架构师改）。
只允许新建或修改：algorithm/lpr_tools/data/、algorithm/lpr_tools/eval/（ablation.py 除外）、algorithm/testset/、
  algorithm/tests/ 下以 test_ccpd、test_manifest、test_eval、test_bench、test_sample、test_golden_pick、test_daily 开头的文件。
禁止：改运行时代码、配置、test_golden.py、scoreboard.md；在测试清单上调参；pip install；切换或合并分支；改其他工位。
协作方式：积极派子代理并行（三项任务互相独立）；看图先缩小到 ≤400 px，一次不超过 8 张。
汇报方式：完整汇报写到 algorithm/out/receipts/第六批.md；聊天里只回一行：回执 algorithm/out/receipts/第六批.md，commit <最后提交号>
```
