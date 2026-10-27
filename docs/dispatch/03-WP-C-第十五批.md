# 派单 03 · 员工 C · 第十五批：把三路新流程设为默认配置（逐张验收、旧流程留档、G1/G7/全量测试）

```text
你是 M2 车牌识别模块执行者【员工 C，工位 03】。工作目录 D:\keshe2\workers\03，分支 feature/m2-wpc-post-eval，用 .venv\Scripts\python.exe。
必读：你第十四批回执；algorithm/config/default.toml、three_route_v2.toml；docs/research/03-三路仲裁集成规格.md 第 7 节；docs/M2-接口说明-M3.md 中配置与 model_version 两处。
背景（架构师 10-28）：
  - 你第十四批已合入 develop（38ab3ac）。架构师用你的 actual.jsonl.gz 对第十三批期望独立复算，3100 张 0 差异；合入后在 develop 抽 100 张重跑，也是 0 差异。
  - 用户决定：新流程现在就设为默认，不等 01 的省份模型。省份模型交付后，只改 default.toml 里的模型路径和阈值，那是下一批的事。
任务：
  1. 切默认：
     a. 把 three_route_v2.toml 里的全部取值写进 default.toml，包括 corner_model_path、box_reader_model_path、arbitration、规格 7.2 的两个浮点阈值、
        两个 solo 开关、nonok_fallback、fallback_predicted_tilt_min_deg。只改取值，不增删键；每个改动的键在行尾注释写明来源（第十三批规格 7.2/7.3）。
        three_route_v2.toml 原样保留作记录。
     b. 新建 algorithm/config/legacy_agree_v1.toml：只写与新默认不同的键，取值等于本批改动前的旧默认（corner_net_v1、box_reader 空、agree、
        agreement_then_reader、各 three_route 键的旧值），供评测报告做“三方法对比/消融”时复现旧流程。
     c. lpr_tools.eval.frozen_once 改为默认读 default.toml，不再需要 --config；正式 flag 仍然不许执行。
  2. 逐张验收（硬指标，必须 0 差异）：
     a. PlateRecognizer.from_config()（不传路径）按固定顺序跑 val1000_v1 + devhard_v1 共 3100 张，与第十三批 main/expected-runtime.jsonl.gz
        比 status、plate、route、候选/rect 索引精确相等，confidence 绝对误差 ≤1e-7。
     b. 用 legacy_agree_v1.toml 跑 val1000_v1，与你第十四批 default-before.jsonl.gz 逐张比 status、plate、全精度 confidence，容差 0。
     有差异就修配置；修不掉的停下写进回执，不许改期望文件。
  3. 其余验证：默认配置下 test_golden 通过并报 20 张里对几张；全量 pytest 跑一次；G7 覆盖率（pytest-cov，lpr 包，≥60%）报总数与各文件；
     G1 用默认配置跑 bench_latency（与第十四批同条件：冻结 320 路径、1280×720、warmup 5、n=100、2 线程），报 P50/P95/最大值与各阶段。
  4. 测试：凡是断言“默认配置是 agree / corner_net_v1 / 不启用框读”的既有测试，改成显式加载 legacy_agree_v1.toml 或显式传参，
     断言内容不许放宽，不许 skip/xfail；每处改动在回执里逐条列出（文件:行、原因）。test_golden.py 不许改。
  5. 文档：docs/M2-接口说明-M3.md 更新默认 model_version 示例与“默认配置”说明（新流程、框读模型、旧流程留档文件名），只改这两处；
     docs/design/ch3-仲裁与评测.md 3.6 节补“已设为默认”一段与本批实测数。
资源与磁盘：不用 GPU；CPU 线程 ≤4（G1 固定 2 线程）；01 正在训练且全机内存紧张，你所有进程合计内存 ≤1 GB，评测一次只跑一个进程；
  新增磁盘 ≤300 MB；pytest 用短 basetemp。临时目录不用清理，回执列出保留了哪些，架构师统一清理。
  预计：改配置与测试 20 分钟；3100 张约 40—60 分钟；val1000 约 10 分钟；全量 pytest + 覆盖率约 15 分钟；G1 约 5 分钟。合计约 1.5—2 小时。
只允许新建或修改：algorithm/config/default.toml（只改上列取值）、algorithm/config/legacy_agree_v1.toml（新建）、algorithm/lpr_tools/eval/frozen_once.py；
  algorithm/tests/ 下按第 4 条需要改的既有测试，以及新建的 test_default_ 开头测试；docs/M2-接口说明-M3.md（只改上列两处）；docs/design/ch3-仲裁与评测.md。
禁止：改 pipeline.py、config.py、arbitration.py 的逻辑（发现缺陷就停下写进回执）；改仲裁规则或阈值；改模型与 manifest、test_golden.py、scoreboard.md；
  在测试清单上跑准确率；pip install；切换或合并分支；改其他工位。
协作方式：可以派子代理并行改测试与文档，但同一时间只跑一个评测进程。
汇报方式：完整汇报写到 algorithm/out/receipts/第十五批.md；聊天里只回一行：回执 algorithm/out/receipts/第十五批.md，commit <最后提交号>
```
