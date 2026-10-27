# 派单 03 · 员工 C · 第十三批：新检测器 + box_reader_v2 组合重算、剩余错误分桶、省份分项、详细设计素材

```text
你是 M2 车牌识别模块执行者【员工 C，工位 03】。工作目录 D:\keshe2\workers\03，分支 feature/m2-wpc-post-eval，用 .venv\Scripts\python.exe。
必读：你第十二批回执与 docs/research/03-三路仲裁集成规格.md；01 第十一批回执（box_reader_v2）；02 第十一批回执（plate_det_v1）；
      algorithm/reports/scoreboard.md 最上面两行；架构师 box_reader_v2 三路复跑 D:\keshe2\algorithm\out\tmp_arch\three_v2\report.json 与 live.jsonl.gz。
背景（架构师 10-28）：
  - develop 已合入 01 box_reader_v2、02 plate_det_v1、你的第十二批。架构师已逐张复核：v2 三路（旧定位 corner_net_v1+传统）majority 回退
    val 97.8%、难例 85.3%、最暗档 48/51、tilt+rotate 88.0%，两集自动通过精度 ≥99.5%。
  - 你第十二批推荐的“自动 011 + 回退 D0_M”是在 box_reader_v1 trace 上选的，换 v2 后已过时（D0_M 在 v1 上最暗档 78.4%，v2 majority 已 94.1%）。
  - plate_det_v1 在难例集 Top1 IoU≥0.7 达 99.5%（旧融合 71%），四角点误差约 6 px（旧 15—20 px）。它不能单独读牌，必须与读牌器组合才看得出整牌效果。
    plate_det_v1 已能用现有入口加载：lpr.recognize:load_corner_detector 按 checkpoint 架构分派（即把 runtime.corner_model_path 指向它）。
  - 省份：CCPD 里 93% 是“皖”；val1000+devhard 中非皖 261 张。架构师在 v2 trace 上算：省份字读对率 框读 90.8%、校正 CRNN 75.5%（33 张读成“皖”）、
    分割+CNN 63.6%（14 张读成“皖”）。两条旧路线一起把外省牌读成“皖”就可能被自动通过——这是本地车牌上的精度风险。
任务：
  1. 组合重算（一条命令，用你的 three_route_workflow --fresh --mode all）：
     a. 主组合：--detector-model algorithm/models/plate_det_v1.pt --detector-loader lpr.recognize:load_corner_detector，
        --box-model algorithm/models/box_reader_v2.pt --box-loader lpr.recognize:load_box_reader --box-pad-x 0.08 --box-pad-y 0.12。
        先确认 detector 换成 plate_det_v1 后，传统定位候选是否仍按 pipeline._merge_candidates 合并进来，回执写明实际候选来源。
     b. 对照：同一 workflow 用旧定位（不传 --detector-model）+ box_reader_v2，可复用架构师 three_v2 trace（--trace），不必重跑推理。
     两个组合都按你第十二批的协议（固定分半、先 fit 选一次、再在 hold 报告；两集合各 ≥99.5%）重新定自动规则与回退规则，并写清是否仍是 011 / D0_M。
     报告：val、绿牌、难例、最暗档、tilt+rotate、各子集，以及按测试集组成折算（清晰=base120+green30、夜间=最暗档、倾斜=tilt35+rotate15、污损=weather）。
  2. 子组自动通过精度：单列最暗档、绿牌、fn、challenge、非皖牌的自动通过精度与张数。若有子组低于 99%，给出一条预先写定、不读真值的补救规则
     （例如“省份字只有一条路线支持时不自动通过”），在 fit 上定、hold 上报，代价（少通过多少张）写清。不许在看过 hold 后再加规则。
  3. 剩余错误分桶（主组合）：每张读错的图归入一类——无候选 / 框偏（IoU<0.7）/ 三路都读错 / 有路线读对但仲裁挑错 / 只错省份字 / 只错其他位 / 8 位牌位数错等，
     按子集与省份列表；每类挑 ≤8 张缩到 ≤400 px 做一张拼图放 out。结论写“下一批训练该补哪类样本”，供架构师派 01、02。
  4. 省份分项：在主组合与对照组合上，按“皖 / 非皖”和每个省份列整牌与省份字读对率、各路线把外省读成“皖”的张数。
  5. 集成规格 v2：把主组合最终推荐的自动规则 + 回退规则（含第 2 项若采纳的补救规则）写成新一节加到 docs/research/03-三路仲裁集成规格.md
     （伪代码、完整浮点阈值、遍历顺序、平分规则、需新增的配置键），配 3100 张逐张期望输出（压缩 JSONL：path、status、plate、confidence、route），供架构师实现后逐张核对。
  6. 《详细设计》第 3 章素材（10-30 交，架构师整理）：新建 docs/design/ch3-仲裁与评测.md——三路仲裁流程（含一张 mermaid 流程图）、配置键与阈值、
     数据划分与五份保护名单排除规则、G2 口径与“自动通过精度”定义、开发集结果表。书面中文，约 2—3 页，数字只引用已复核的。
资源与磁盘：不用 GPU；CPU 线程 ≤4；你所有进程合计内存 ≤1 GB，评测一次只跑一个进程；新增磁盘 ≤500 MB，trace 用压缩 JSONL；pytest 临时目录跑完就删。
  预计：第 1 项 a 实跑 3100 张约 30—60 分钟，其余为离线计算。
只允许新建或修改：algorithm/lpr_tools/eval/（ablation.py 除外）；algorithm/tests/ 下以 test_eval、test_three_route、test_fallback 开头的文件；
  docs/research/03-三路仲裁集成规格.md；docs/design/ch3-仲裁与评测.md。
禁止：改运行时代码（algorithm/lpr/）、配置、模型、test_golden.py、scoreboard.md；在测试清单上推理或调参；pip install；切换或合并分支；改其他工位；删除本批自己临时文件以外的任何文件。
协作方式：积极派子代理，但同一时间只跑一个评测进程；看图先缩小到 ≤400 px，一次不超过 8 张。
汇报方式：完整汇报写到 algorithm/out/receipts/第十三批.md；聊天里只回一行：回执 algorithm/out/receipts/第十三批.md，commit <最后提交号>
```
