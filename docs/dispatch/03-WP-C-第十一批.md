# 派单 03 · 员工 C · 第十一批：“检测框直接读”的零训练起点、三路仲裁工具、磁盘清理清单

```text
你是 M2 车牌识别模块执行者【员工 C，工位 03】。工作目录 D:\keshe2\workers\03，分支 feature/m2-wpc-post-eval，用 .venv\Scripts\python.exe。
必读：docs/M2-架构师交接.md 第 1 节与第 6 节 0''' 条（10-27 更正）；你第十批回执；algorithm/lpr/boxcrop.py（crop_box，新契约）；
      algorithm/lpr/pipeline.py 当前默认流程（agree + agreement_then_reader）；D:\keshe2\docs\dispatch\01-WP-A-第十批.md 与 02-WP-B-第十一批.md（了解另外两人在做什么）。
背景（架构师 10-27，用户已拍板）：
  - 新路线：增加“检测框外扩后直接读”。参考系统（PMC11086086：YOLOv5 + PDLPR，不做校正）在 CCPD 上 DB 99.5%、Tilt 99.3%。
    01 本批训练 box_reader_v1，02 本批训练检测器 plate_det_v1。之后流水线每个候选会有三种读法：分割+CNN、CRNN 读校正裁片、box_reader 读框裁片。
  - 用户提醒：本项目已占 100 多 GB 磁盘，不能再无限涨。
任务：
  1. 零训练起点（先做，尽快出数）：用现有 corner_net_v1 + 传统定位的 Top-3 候选框，以及真值外接框，按 crop_box 外扩若干组
     （pad_x ∈ {0.05, 0.10, 0.15, 0.25}，pad_y ∈ {0.15, 0.25, 0.40}），用 crnn_v2 直接读，报 val1000_v1、devhard_v1 各子集整牌准确率
     （单列最暗档：文件名亮度 ≤19，以及 tilt+rotate），与 crnn_v2 读校正裁片（ceiling_v2 报告）对比。这是“框直接读”在不训练时的起点。
  2. 三路 trace 与仲裁模拟工具 lpr_tools/eval/three_route.py：对每个候选记录三种读法（box 路线的模型与 box_pad 做成参数，现在先用 crnn_v2 + 第 1 项最好的一组外扩代替），
     沿用第九、十批的精确扫描与真实复跑核对思路，模拟几种仲裁规则（例如三取二一致才自动通过、box 路线单边高置信通过、不通过时回退取哪个读法），
     约束 val1000_v1 与 devhard_v1 自动通过精度同时 ≥99.5%，在此约束下自动通过数最多，同时报整牌准确率（分子集）。
     01、02 的模型交回后要能一条命令重跑。只推荐规则，流水线由架构师改。
  3. 磁盘清理清单（只读，不删除任何东西）：统计 D:\keshe2\algorithm\out、D:\keshe2\algorithm\data（raw 除外）、D:\keshe2\workers\0X\algorithm\out 下各子目录大小，
     分成“可删（可再生的中间产物：pytest 临时目录、中间 checkpoint、重复评测输出、批量缩略图、已被新版替代的字符/裁片数据集等）”“建议保留”“必须保留（被回执、模型登记或当前训练引用）”三档，
     每项写路径、大小、理由，写到 algorithm/out/receipts/第十一批-磁盘清单.md。统计要轻量，一次只统计一个目录，不要并发大量扫描。
资源与磁盘（这是用户的机器，内存紧张，必须遵守）：不用 GPU；CPU 线程 ≤4；你所有进程合计内存 ≤1 GB（评测一次只跑一个进程）；本批新增磁盘 ≤1 GB，trace 用压缩 JSONL；pytest 临时目录跑完就删。
只允许新建或修改：algorithm/lpr_tools/data/、algorithm/lpr_tools/eval/（ablation.py 除外）、algorithm/testset/、
  algorithm/tests/ 下以 test_ccpd、test_manifest、test_eval、test_bench、test_sample、test_golden_pick、test_daily、test_build_plates、test_ceiling、test_crop_gap、test_three_route 开头的文件。
禁止：改运行时代码（algorithm/lpr/）、配置、模型、test_golden.py、scoreboard.md；删除本批自己产生的临时文件以外的任何文件（清单只列不删）；在测试清单上调参；pip install；切换或合并分支；改其他工位。
协作方式：积极派子代理，但同一时间只跑一个评测进程；看图先缩小到 ≤400 px，一次不超过 8 张。
汇报方式：完整汇报写到 algorithm/out/receipts/第十一批.md；聊天里只回一行：回执 algorithm/out/receipts/第十一批.md，commit <最后提交号>
```
