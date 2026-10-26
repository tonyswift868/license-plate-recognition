# 派单 01 · 员工 A · 第八批：先调研同类项目，再把整牌识别训练成难例主力

```text
你是 M2 车牌识别模块执行者【员工 A，工位 01】。工作目录 D:\keshe2\workers\01，分支 feature/m2-wpa-model，用 .venv\Scripts\python.exe。
必读：你第七批回执；algorithm/reports/scoreboard.md 最新一行；D:\keshe2\algorithm\data\plates\v2\stats.md 与 handoff.json（03 做的难例整牌数据）；
      docs/M2-车牌识别-开发文档.md 第 2 章（C2、C3）、14 章 K1；algorithm/lpr/pipeline.py 的 _arbitrate 与 default.toml [postprocess] 新增的 agree 模式。
背景（架构师 10-26 判断）：
  - 难例上“先切字再逐字认”已经到顶：给真值四角点，分割+CNN 在难例开发集也只有 56%，2100 张里 519 张切不开字。这条路不再调。
  - 同类项目在 CCPD 上都不切字：LPRNet、RPnet 一类整牌识别，论文里 DB（夜间）约 97—98%、Tilt 约 93—99%。所以难例主力换成整牌识别。
  - 流水线已加 agree 模式：分割路线与整牌识别读法一致才自动通过，不一致转人工。整牌识别不再只是兜底，它的准确率和置信度直接决定难例成绩。
任务：
  0. 先调研同类项目（2 小时内，派子代理并行；同时可以先准备数据管线）：
     至少看 LPRNet（arXiv 1806.10447）、CCPD 官方 RPnet（github.com/detectRecog/CCPD）、LPTR-AFLNet（arXiv 2507.16362，有 CCPD 分子集对比表）、
     CRPD（arXiv 2205.03582），再找 2—3 个 GitHub 上用 CCPD 训练的中文车牌识别项目（YOLO+LPRNet、CRNN+CTC 一类）。不能联网就凭已有知识写，并注明哪些数字未核实。
     写 docs/research/01-整牌识别调研.md：每个方案的网络结构、输入尺寸、模型大小、CPU 速度、CCPD 分子集准确率（重点 DB、Tilt、Blur）、
     训练数据与增广技巧（尤其低光照、模糊、倾斜）、对我们是否适用（CPU、从零训练、≤5 MB、不能用预训练权重）。
     结尾写清楚：我们采用哪些做法及理由；哪些我们已经在做；哪些做不了。借鉴的只是思路和结构，代码自己写、权重自己从零训练，文档注明出处。
  1. 按调研结论训练整牌识别 v2：数据用 plates/v2（训练集已含难例训练池）。可以在 crnn_v1 上续训，也可以按调研换成 LPRNet 类结构从零训练，回执写选择理由。
     先从 plates/v2 train 里按原图留出“难例调参集”（7 个难例子集各 300 张原图，它们的所有裁片都不进训练）；选轮次、调参只用它和 plates/v2 val；
     D:\keshe2\algorithm\data\index\devhard_v1.csv 只在最后报告用。
     置信度要校准（例如温度缩放），让“置信度高就基本读对”成立；回执给出难例调参集上按置信度分档的精度表。
     模型存 algorithm/models/crnn_v2.pt（≤5 MB，入库），用 write_manifest_entry 登记（只加你的条目）。训练放后台，CPU 训练线程 ≤8。
  2. 评测：
     a. 读牌上限：难例调参集真值四角点裁片的整牌准确率，7 个子集分开报，并单列夜间最暗一档（CCPD 文件名亮度字段 ≤19，与测试集夜间口径一致）。目标：每个子集 ≥95%。
     b. 整帧：LPR_CONFIG 临时配置（CNN v2 + corner_net_v1 + crnn_v2），在 val1000_v1 与 devhard_v1 上分别跑 postprocess.arbitration = "first_ok" 和 "agree"（agree 阈值用默认值），
        与 crnn_v1 对比整牌准确率、自动通过率、自动通过精度，按子集分开。正式阈值由 03 扫描，你不用定。
     c. 耗时：单块 read() ≤10 ms（CPU 2 线程），G1 整帧 P95 ≤200 ms。
  止损：某个子集训练到不再上升时仍明显 <90%，不要继续磨。停下来在回执写清楚哪类图读不出（附 ≤8 张缩略图）、你判断的原因、调研里别人怎么处理这类图。
只允许新建或修改：algorithm/lpr/recognize/crnn.py、cnn_model.py、cnn.py、template.py、svm.py；algorithm/lpr_tools/train/；algorithm/lpr_tools/eval/ablation.py；
  algorithm/models/crnn_v2.pt 与 algorithm/models/manifest.json（只加你的条目）；algorithm/tests/ 下以 test_crnn、test_recognize、test_train 开头的文件；
  docs/research/01-整牌识别调研.md；D:\keshe2\algorithm\data\chars\ 与 D:\keshe2\algorithm\data\plates_synth\。
禁止：改 pipeline.py、config.py、recognize/base.py、default.toml；任何预训练权重、现成 OCR 或车牌识别库、复制他人代码、pip install；
      用测试清单、金样例、devhard_v1.csv、val1000_v1.csv、坏图名单里的图训练或调参；切换或合并分支；改其他工位。
协作方式：积极派子代理并行（调研、数据、训练、评测可并行，训练放后台）；看图先缩小到 ≤400 px，一次不超过 8 张。
汇报方式：完整汇报写到 algorithm/out/receipts/第八批.md；聊天里只回一行：回执 algorithm/out/receipts/第八批.md，commit <最后提交号>
```
