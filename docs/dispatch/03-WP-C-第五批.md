# 派单 03 · 员工 C · 第五批：测试集损坏图、评测测试修复、G1 实测、逐阶段损失归因

```text
你是 M2 车牌识别模块执行者【员工 C，工位 03】。工作目录 D:\keshe2\workers\03，分支 feature/m2-wpc-post-eval，用 .venv\Scripts\python.exe。
必读：algorithm/reports/scoreboard.md 第一行；D:\keshe2\algorithm\out\day1\val\report.json、hard\report.json；docs/M2-车牌识别-开发文档.md 7.4、12.5、13 章。
现状：develop 已切到 CNN v0（default.toml 的 runtime.model_path），首次端到端整牌准确率只有 2.4%。
任务：
  1. 损坏图：测试清单里 D:\keshe2\algorithm\data\raw\ccpd2019\ccpd_np\5125.jpg 只有 4096 字节、无法解码。
     扫描测试清单 320 张、金样例 20 张、以及整个索引里的图片能否解码；把整个索引的不可解码图列到
     D:\keshe2\algorithm\data\index\undecodable.csv。测试清单按原规则（seed 42、同类别、来自 ccpd_np test）补一张替换，
     生成 v1.1（manifest.csv 与共享副本都更新，v1.json 记录变更原因和被替换的图）；sample_testset、eval_accuracy、bench_latency
     遇到不可解码图要跳过并计数，不能整体报错退出。
  2. develop 合并后 tests/test_eval_accuracy.py::test_cli_and_scoreboard 失败，且单个测试耗时约 184 秒（疑似读了真实共享数据或真实模型）。
     查清原因修好；所有单测必须只用 tmp 里构造的数据，不读 D:\keshe2\algorithm\data\ 和真实模型，全套测试保持在几十秒内。
  3. G1 实测：用 CNN v0 在测试清单（v1.1）上跑 bench_latency（1280×720，预热 5 次，计时 100 次），给出 P50/P95/最大值与各阶段耗时。
  4. 逐阶段损失归因：在同一批 val 1000 张（eval_accuracy --split val --n 1000 --seed 42）上，分别跑正常模式和 --oracle 模式，
     做一张表：定位损失、校正+分割损失、识别损失各占多少；识别错误给出最常见的 20 个“真值字符 → 识别字符”混淆对和出错位次分布。
     结论写清“下一步该先修哪个阶段”，给 01、02 号员工参考。
只允许新建或修改：algorithm/lpr_tools/data/、algorithm/lpr_tools/eval/（ablation.py 除外）、algorithm/testset/、
  algorithm/tests/ 下以 test_ccpd、test_manifest、test_eval、test_bench、test_sample、test_golden_pick 开头的文件、D:\keshe2\algorithm\data\index\ 下的文件（共享索引 ccpd_index.csv 本身不改）。
禁止：改其他代码、配置、tests/test_golden.py、scoreboard.md；在测试清单上调任何参数；pip install；切换或合并分支；改其他工位。
协作方式：积极派子代理并行（第 1、2 项与第 3、4 项可并行）；看图先缩小到 ≤400 px，一次不超过 8 张。
汇报方式：完整汇报写到 algorithm/out/receipts/第五批.md；聊天里只回一行：回执 algorithm/out/receipts/第五批.md，commit <最后提交号>
```
