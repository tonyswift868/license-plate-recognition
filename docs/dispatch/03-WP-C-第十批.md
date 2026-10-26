# 派单 03 · 员工 C · 第十批：crnn_v2 下重定 agree 阈值，拆解“找到了却读错”

```text
你是 M2 车牌识别模块执行者【员工 C，工位 03】。工作目录 D:\keshe2\workers\03，分支 feature/m2-wpc-post-eval，用 .venv\Scripts\python.exe。
必读：你第九批回执；algorithm/reports/scoreboard.md 最新两行；D:\keshe2\workers\01\algorithm\out\receipts\第八批.md（crnn_v2）；
      D:\keshe2\workers\02\algorithm\out\receipts\第九批.md（角点误差诊断：db 上“框命中但角点平均偏 >12 px”占多数）；docs/M2-车牌识别-开发文档.md 5.4、7.4 节。
背景（架构师 10-27）：
  - crnn_v2 已合入（难例训练）。难例开发集整牌 53.6%→68.1%，但测试集口径的“夜间”（db 亮度 ≤19）只有约 43%，倾斜约 74%，离 85% 还远。
  - 你的上限矩阵显示最暗档 Top-3 定位命中 48/51；01 的调参集上 crnn_v2 读真值裁片，最暗档 86%。也就是说车牌大多找到了、正确裁出来也大多读得对，
    丢分集中在“找到了却读错”这一段。要把这一段拆清楚，下一步才知道该改角点、改读牌还是改候选挑选。
  - 架构师用 ceiling.py 跑过 crnn_v2 的矩阵，结果在 D:\keshe2\algorithm\out\tmp_arch\ceiling_v2\report.json，可直接引用，不必重跑。
    最暗档：真值四角点读 82.4%，实际候选只用 CRNN 读 58.8%，但整条流水线（first_ok）只有 43.1%、agree 默认阈值 33.3%——
    说明除了角点不准，流水线“在两条路线、多个候选之间挑哪个读法返回”也在丢分。
任务：
  0. 调研（1 小时内，派子代理）：同类系统怎么在多个候选框、多种裁法之间挑最终结果（多裁片测试时增广 TTA、按置信度或投票选、检测分与识别分联合排序等），
     写 docs/research/03-候选与多裁片选择调研.md，结尾写哪些能直接用到我们的流水线。不能联网就凭已有知识写，并注明未核实处。
  1. agree 阈值重扫：配置 CNN v2 + corner_net_v1 + crnn_v2（LPR_CONFIG 临时配置），沿用第九批的精确扫描与真实复跑核对，
     约束 val1000_v1 与 devhard_v1 自动通过精度同时 ≥99%（另给同时 ≥99.5% 的保守档），给出推荐阈值、整牌准确率（分子集，另列 db 亮度 ≤19 一档）、自动通过率、G1 P95。
     这一档将作为下一版默认配置的候选，只推荐，不写 default.toml。
     另外用同一份 trace 模拟比较“没有任何结果能自动通过时，返回哪个读法”的几种规则（现行 _rank：先状态后置信度跨路线比较；第一个候选的 CRNN 读法；
     所有候选里 CRNN 置信度最高的读法；两路线一致优先、否则 CRNN 等），在不改变自动通过集合的前提下，选整牌准确率最高的规则，
     分子集报（单列最暗档、tilt+rotate、val 的 green）。只推荐规则，流水线由架构师改。
  2. “找到了却读错”分解：在 devhard_v1 的 db（并单列亮度 ≤19）、tilt、rotate 上，对“Top-3 有候选命中（IoU ≥0.5）但最终读错”的每张图，离线比较以下读法（只做诊断，不改运行时）：
     a. 流水线最终用的那块裁片；b. Top-3 每个候选各自的裁片（crnn_v2 与分割+CNN 都读）；c. 命中候选的 bbox 外扩 0/10/20% 后不做透视、直接缩放的裁片；d. 真值四角点裁片。
     按互斥桶统计：挑错候选（别的候选能读对）、角点不准（真值裁片能读对而所有候选裁片都读错）、外扩裁片就能救回、读牌本身读不对（真值裁片也错）、颜色误判成特种牌返回 UNSUPPORTED 等。
     每个桶给数量与 ≤8 张缩略图（≤400 px）。结论写清楚：最该改的是角点、读牌还是候选挑选，各能救回多少张。
  3. 把第 2 项做成一条命令可重跑的工具（lpr_tools/eval/crop_gap.py），模型路径做成参数，后面新模型交回后能直接复测。
止损：第 2 项若发现某一桶占绝对多数，先把这一桶的数字和样例写进回执，不必把其余细分做满。
只允许新建或修改：algorithm/lpr_tools/data/、algorithm/lpr_tools/eval/（ablation.py 除外）、algorithm/testset/、
  algorithm/tests/ 下以 test_ccpd、test_manifest、test_eval、test_bench、test_sample、test_golden_pick、test_daily、test_build_plates、test_ceiling、test_crop_gap 开头的文件；
  docs/research/03-候选与多裁片选择调研.md；D:\keshe2\algorithm\data\plates\。
禁止：改运行时代码（algorithm/lpr/）、配置、模型、test_golden.py、scoreboard.md；在测试清单上调参；pip install；切换或合并分支；改其他工位。
协作方式：积极派子代理并行（调研、阈值重扫、分解可同时做）；看图先缩小到 ≤400 px，一次不超过 8 张。
汇报方式：完整汇报写到 algorithm/out/receipts/第十批.md；聊天里只回一行：回执 algorithm/out/receipts/第十批.md，commit <最后提交号>
```
