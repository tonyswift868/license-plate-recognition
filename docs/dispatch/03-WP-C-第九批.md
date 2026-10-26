# 派单 03 · 员工 C · 第九批：夜间上限实验、agree 模式 E1 阈值、自动通过判定调研

```text
你是 M2 车牌识别模块执行者【员工 C，工位 03】。工作目录 D:\keshe2\workers\03，分支 feature/m2-wpc-post-eval，用 .venv\Scripts\python.exe。
必读：你第八批回执；algorithm/reports/scoreboard.md 最新一行；docs/M2-车牌识别-开发文档.md 5.4 节、7.4 节；
      algorithm/lpr/pipeline.py 的 _run 与 _arbitrate；algorithm/config/default.toml [postprocess] 新增五项；D:\keshe2\workers\01\algorithm\out\receipts\第七批.md（crnn_v1 结果）。
背景（架构师 10-26）：
  - 流水线新增 agree 模式（默认仍是 first_ok）：同一块校正图上，分割路线与 CRNN 读法相同、两边置信度较小值 ≥ agree_min_confidence → OK；
    不一致时，单边置信度 ≥ solo_min_confidence_segment / solo_min_confidence_reader 才 OK，否则 LOW_CONFIDENCE，备选里有两边读法；返回哪边读法由 disagree_prefer 决定。
  - crnn_v1 已合入。开 CRNN 后 val 整牌 95.7%，但旧规则下自动通过精度 val 98.6%、难例 94.5%，不能直接上线。
  - 测试集“夜间”50 张是 ccpd_db 里亮度字段 ≤19 的最暗一档（开发集 db 中位 29），按开发集估算现在只有约 29%。要先弄清楚丢分在“找不到车牌”还是“找到了读不出”，再决定投不投入。
任务：
  0. 调研（1 小时内，派子代理）：同类车牌识别 / OCR 系统怎么决定“自动放行还是转人工”：多模型一致性、置信度校准（温度缩放等）、拒识阈值怎么定、报告里怎么写。
     写 docs/research/03-自动通过判定调研.md，结尾写哪些做法可以直接用到 agree 模式与 E1 阈值上。不能联网就凭已有知识写，并注明哪些未核实。
  1. 上限矩阵（夜间上限实验）：写 lpr_tools/eval/ceiling.py，一条命令在指定名单上输出 2×2 矩阵：
     {真值四角点, 实际定位} × {分割+CNN, 整牌识别 CRNN} 的整牌准确率，按子集分开，并单列 db 亮度 ≤19 一档；另报实际定位 Top-3 是否命中。
     模型路径（CNN、corner、CRNN）都做成参数。用 val1000_v1、devhard_v1 与当前 crnn_v1 跑一遍出结果；01 的 crnn_v2、02 的 corner_net_v3 交回后要能一条命令重跑。
     结论写清楚：夜间、倾斜各自的丢分里，多少是“找不到车牌”，多少是“找到了读不出”。
  2. agree 模式 E1 阈值：配置 CNN v2 + corner_net_v1 + crnn_v1，在 val1000_v1 与 devhard_v1 上记录每块校正图两条路线的结果，
     扫描 agree_min_confidence、两个 solo 门槛和 disagree_prefer。约束：val 与 devhard 的自动通过精度同时 ≥99%（另给一档同时 ≥99.5% 的保守值），
     在此约束下自动通过率最高；同时报整牌准确率（分子集），并和旧规则 first_ok 对比。
     可以用 trace 模拟扫描，但选出的每一档都要用真实流水线（LPR_CONFIG 临时配置）复跑，逐张核对状态、读法、置信度一致。只推荐，不写 default.toml。
  3. daily.py 支持指定 agree 配置运行，产出一行积分板（不改 scoreboard.md）；agree 模式每块校正图都会调用 CRNN，要单独报 G1 P95（≤200 ms）。
只允许新建或修改：algorithm/lpr_tools/data/、algorithm/lpr_tools/eval/（ablation.py 除外）、algorithm/testset/、
  algorithm/tests/ 下以 test_ccpd、test_manifest、test_eval、test_bench、test_sample、test_golden_pick、test_daily、test_build_plates、test_ceiling 开头的文件；
  docs/research/03-自动通过判定调研.md；D:\keshe2\algorithm\data\plates\。
禁止：改运行时代码（algorithm/lpr/）、配置、test_golden.py、scoreboard.md；在测试清单上调参；pip install；切换或合并分支；改其他工位。
协作方式：积极派子代理并行（调研、上限矩阵、E1 扫描可同时做）；看图先缩小到 ≤400 px，一次不超过 8 张。
汇报方式：完整汇报写到 algorithm/out/receipts/第九批.md；聊天里只回一行：回执 algorithm/out/receipts/第九批.md，commit <最后提交号>
```
