# 派单 02 · 员工 B · 第十批：角点精修（只管“框对了但角点偏”）

```text
你是 M2 车牌识别模块执行者【员工 B，工位 02】。工作目录 D:\keshe2\workers\02，分支 feature/m2-wpb-locate，用 .venv\Scripts\python.exe。
必读：你第九批回执与 docs/research/02-车牌定位调研.md；algorithm/reports/scoreboard.md 最新两行；
      D:\keshe2\workers\01\algorithm\out\receipts\第八批.md（crnn_v2：真值裁片 85.95%，流水线候选裁片 70.60%）；
      架构师用 crnn_v2 跑的上限矩阵 D:\keshe2\algorithm\out\tmp_arch\ceiling_v2\report.json。
背景（架构师 10-27）：
  - 你第九批的诊断很关键：db 融合候选里“框命中但四角点平均偏 >12 px”有 181/300，最暗档 38/51；召回其实不差（最暗档 Top-3 48/51）。
    架构师用 ceiling.py 在 devhard_v1 同一批图上测了 crnn_v2：用真值四角点裁片读，最暗档 82.4%、tilt 94.0%、rotate 95.7%；
    用实际候选裁片读，只有 58.8%、68.0%、83.7%。角点做准，这几类能直接涨 12—26 个百分点，这是目前最大的一块。
    所以本批不再追召回，只解决“角点不准导致裁片不准”。corner_net_v3 不采用，召回方向本批不做。
  - 已证明的结论照用：零训练的亮度增强收益很小，不再试。
任务：
  0. 调研（1 小时内，派子代理）：同类方案怎么把角点做准——WPOD-NET 的局部仿射、LPTR-AFLNet 的 PTR 透视校正器、两级“先粗后精”的 ROI 角点回归、
     在候选裁片上用高分辨率热图或边缘线拟合求角点等。补充到 docs/research/02-车牌定位调研.md 新的一节，结尾写本批采用什么、为什么。
  1. 做一个独立的“角点精修器”：输入是任一来源候选（corner_net_v1、传统定位）外扩后的局部裁片，输出精修后的四角点；
     训练用难例训练池与 base/green（排除四份名单：testset_manifest_v1、golden_manifest、devhard_v1、undecodable，路径与同字节都排除），
     输入框用真值框加上与真实候选误差分布相当的随机偏移来模拟；调参只用你的独立 holdout，devhard_v1 只在最后报告。
     模型存 algorithm/models/corner_refine_v1.pt（≤5 MB），登记 manifest（只加你的条目）。精修器的推理代码写在 corner_net.py 里，提供一个可单独调用的函数或类，不接入流水线（接入由架构师做）。
  2. 评测：
     a. 在 devhard_v1 的 db（单列亮度 ≤19）、tilt、rotate 上，对 bbox 命中的候选，报精修前后四角点平均误差与 ≤8 px 比例。目标：平均误差 ≤8 px。
     b. 用精修后的四角点做透视裁片，再用 crnn_v2 读（离线评测，不改 pipeline.py），报这些子集的整牌准确率与精修前对比；val1000_v1 上同样做一遍，不得变差。
        目标：最暗档 ≥75%、tilt ≥85%（精修前分别为 58.8%、68.0%，真值四角点上限 82.4%、94.0%）。
     c. 精修器单次调用 ≤15 ms（CPU 2 线程）。
  止损：训练两轮后 holdout 上平均角点误差下降不到 30%，就停下，写清原因和调研里别人的做法，不要继续磨同一结构。
只允许新建或修改：algorithm/lpr/corner_net.py、preprocess.py、locate.py、rectify.py、segment.py；algorithm/lpr_tools/train_corner.py、synth_ccpd.py、calib_layout.py、viz.py，
  以及新建 algorithm/lpr_tools/train_refine.py、algorithm/lpr_tools/eval_refine.py；
  algorithm/models/corner_refine_v1.pt 与 algorithm/models/manifest.json（只加你的条目）；algorithm/tests/ 下以 test_corner、test_refine、test_preprocess、test_locate、test_rectify、test_segment 开头的文件；
  docs/research/02-车牌定位调研.md；D:\keshe2\algorithm\data\corner\ 与 D:\keshe2\algorithm\data\calib\。
禁止：改 pipeline.py、config.py、recognize/、default.toml；任何预训练权重、现成检测模型或车牌识别库、复制他人代码、pip install；
      用四份名单里的图训练或调参；切换或合并分支；改其他工位。
协作方式：积极派子代理并行（调研、数据、训练、评测可并行，训练放后台，训练线程 ≤8）；看图先缩小到 ≤400 px，一次不超过 8 张。
汇报方式：完整汇报写到 algorithm/out/receipts/第十批.md；聊天里只回一行：回执 algorithm/out/receipts/第十批.md，commit <最后提交号>
```
