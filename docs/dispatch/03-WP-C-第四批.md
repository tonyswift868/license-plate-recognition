# 派单 03 · 员工 C · 第四批：测试集 v1 冻结、金样例入库、第一批真实指标

```text
你是 M2 车牌识别模块执行者【员工 C，工位 03】。工作目录 D:\keshe2\workers\03，分支 feature/m2-wpc-post-eval，用 .venv\Scripts\python.exe。
必读：docs/M2-车牌识别-开发文档.md 7.4 节（V0.5 已改：ccpd_base 没有 test 划分，清晰测试图从 base 验证集留出 120 张）、
      12.5 节、13 章；docs/M2-数据到位运行手册.md
共享数据（不入库，绝对路径）：原始 D:\keshe2\algorithm\data\raw\，索引 D:\keshe2\algorithm\data\index\ccpd_index.csv。
本批允许写入共享目录：D:\keshe2\algorithm\data\index\ 下新增文件。
任务（按顺序，第 1 项最急，01 号员工在等它）：
  1. 测试集 v1：改 sample_testset.py 支持“清晰 = ccpd_base val 按种子 42 留出 120 + CCPD2020 test 30”；夜间按 ccpd_db 亮度分布
     定阈值（在回执里给出分布和理由）；倾斜 50；污损 50 从候选里看缩略图挑真正污损/遮挡的（每次看不超过 8 张缩到 ≤400 px 的图，
     note 写“AI 目检，待tonyswift868复核”）；无牌 20。输出正式 algorithm/testset/manifest.csv（入库），并复制一份到
     D:\keshe2\algorithm\data\index\testset_manifest_v1.csv。完成后立刻提交。
  2. eval_accuracy 的 --split val 默认剔除测试清单里的图（参数 --exclude-manifest，默认指向上面的共享副本），保证调参不碰测试集。
     约定“难例开发集”= ccpd_db/blur/tilt/rotate/fn/challenge/weather 中不在测试清单里的图，供调难例用；给 eval_accuracy 加 --dev-hard 选项按子集分层抽样。
  3. 金样例：按开发文档 12.1 的构成挑 20 张（base 的 10 张从留出的清晰测试图里挑），写入 algorithm/tests/data/golden/ 与 labels.csv（本批授权写入，入库），
     test_golden.py 的门槛不要改。
  4. 第一批真实指标（不需要模型）：在 ccpd_base val 1000 张（剔除测试清单）上测 Top-3 定位召回；在难例开发集每个子集各 300 张上测定位召回；
     用 --uniform 跑 bench_latency（1280×720，100 次）。只写回执，不改积分板。
只允许新建或修改：algorithm/lpr_tools/data/、algorithm/lpr_tools/eval/（ablation.py 除外）、algorithm/testset/、algorithm/tests/data/golden/、
  algorithm/tests/ 下以 test_ccpd、test_manifest、test_eval、test_bench、test_sample、test_golden_pick 开头的文件、D:\keshe2\algorithm\data\index\ 下新增文件。
禁止：改其他代码、配置、tests/test_golden.py、scoreboard.md；在测试清单的图上调任何参数；pip install；切换或合并分支；改其他工位。
协作方式：积极派子代理并行（如第 2、4 项与第 3 项并行）；看图先缩小，一次不超过 8 张。
汇报方式：完整汇报写到 algorithm/out/receipts/第四批.md；聊天里只回一行：回执 algorithm/out/receipts/第四批.md，commit <最后提交号>
```
