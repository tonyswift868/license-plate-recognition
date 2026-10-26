# 派单 02 · 员工 B · 第九批：先调研同类项目的定位做法，再攻夜间与倾斜定位

```text
你是 M2 车牌识别模块执行者【员工 B，工位 02】。工作目录 D:\keshe2\workers\02，分支 feature/m2-wpb-locate，用 .venv\Scripts\python.exe。
必读：你第八批回执（含 db holdout 的 coarse/ROI 漏检诊断）；algorithm/reports/scoreboard.md 最新一行；docs/M2-车牌识别-开发文档.md 第 2 章、7.4 节、14 章 K1。
背景（架构师 10-26 判断）：
  - 你第八批的 corner_net_v2 不采用（比 v1 差），已作为实验记录合入。同一结构换数据重训这条路不再继续。
  - corner_net_v1 在 devhard_v1 的 Top-3 定位召回：db 81%、fn 84%、tilt 87%、blur 87%。测试集“夜间”50 张是 ccpd_db 里最暗的一档（文件名亮度字段 ≤19），
    车牌找不到，后面识别再准也到不了 85%。同类项目（CCPD 官方 RPnet、WPOD-NET、YOLO 加四角点关键点一类）在 DB 子集的检测都在 95% 以上，说明差距在方法。
任务：
  0. 先调研同类项目（2 小时内，派子代理并行）：至少看 CCPD 论文与 RPnet 的检测部分（github.com/detectRecog/CCPD）、
     WPOD-NET（Silva & Jung，ECCV 2018，github.com/sergiomsilva/alpr-unconstrained）、CRPD（arXiv 2205.03582，四角点检测+识别一体），
     再找 2—3 个 GitHub 上“YOLO + 车牌四角点关键点”的中文车牌项目，以及低光照车牌检测的做法（输入增强还是训练增广）。不能联网就凭已有知识写，并注明哪些数字未核实。
     写 docs/research/02-车牌定位调研.md：每个方案的检测方式（框回归、关键点热图、仿射参数等）、输入尺寸、模型大小、CPU 速度、CCPD 分子集检测结果、
     处理夜间与倾斜的技巧、对我们是否适用（CPU、从零训练、≤5 MB、不能用预训练权重或现成检测模型）。
     结尾写清楚：我们的两级四角点网络和它们差在哪里，下一步采用什么。借鉴的只是思路和结构，代码自己写、权重自己从零训练，文档注明出处。
  1. 夜间定位诊断（不训练，先做）：在 devhard_v1 的 db 300 张及其中亮度 ≤19 的一档上，分别统计 corner v1、传统定位、两者融合后的 Top-1/Top-3 召回（IoU ≥0.5 与 ≥0.7），
     失败分成三类：没有任何候选；有候选但框偏；框对但四角点偏（报角点平均像素误差）。
     再试零训练的办法：送入网络前做亮度增强（gamma、CLAHE 等），暗图做两次检测取并集等。给出每种办法的召回变化与耗时。
  2. 按调研结论改进定位，出 corner_net_v3：训练用难例训练池（排除四份名单：testset_manifest_v1、golden_manifest、devhard_v1、undecodable，路径与同字节都排除）；
     调参只用你第八批的独立 holdout，devhard_v1 只在最后报告。
     目标：devhard db Top-3 ≥92%、亮度 ≤19 一档 ≥90%、tilt ≥93%；val1000_v1 整牌不低于 v1 的 85.3%；G1 P95 ≤200 ms。
     模型存 algorithm/models/corner_net_v3.pt（≤5 MB，入库），登记 manifest.json（只加你的条目）。亮度增强等前处理如需上线，写在 corner_net.py 内部。
  止损：第 1 项如果零训练办法已经把召回拉到目标附近，先把它做成正式实现，不必硬训 v3；v3 训练两轮仍不如 v1，就停下，写清原因和调研里别人的做法，不要继续磨同一结构。
只允许新建或修改：algorithm/lpr/corner_net.py、preprocess.py、locate.py、rectify.py、segment.py；algorithm/lpr_tools/train_corner.py、synth_ccpd.py、calib_layout.py、viz.py；
  algorithm/models/corner_net_v3.pt 与 algorithm/models/manifest.json（只加你的条目）；algorithm/tests/ 下以 test_corner、test_preprocess、test_locate、test_rectify、test_segment 开头的文件；
  docs/research/02-车牌定位调研.md；D:\keshe2\algorithm\data\corner\ 与 D:\keshe2\algorithm\data\calib\。
禁止：改 pipeline.py、config.py、recognize/base.py、default.toml；任何预训练权重、现成检测模型或车牌识别库、复制他人代码、pip install；
      用四份名单里的图训练或调参；切换或合并分支；改其他工位。
协作方式：积极派子代理并行（调研与第 1 项可同时做，训练放后台，训练线程 ≤8）；看图先缩小到 ≤400 px，一次不超过 8 张。
汇报方式：完整汇报写到 algorithm/out/receipts/第九批.md；聊天里只回一行：回执 algorithm/out/receipts/第九批.md，commit <最后提交号>
```
