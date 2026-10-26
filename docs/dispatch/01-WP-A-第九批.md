# 派单 01 · 员工 A · 第九批：让整牌识别扛得住“裁得不准”的车牌

```text
你是 M2 车牌识别模块执行者【员工 A，工位 01】。工作目录 D:\keshe2\workers\01，分支 feature/m2-wpa-model，用 .venv\Scripts\python.exe。
必读：你第八批回执与 docs/research/01-整牌识别调研.md；algorithm/reports/scoreboard.md 最新两行；
      D:\keshe2\workers\02\algorithm\out\receipts\第九批.md（角点误差诊断表）；D:\keshe2\workers\03\algorithm\out\receipts\第九批.md（上限矩阵）；
      架构师用 crnn_v2 跑的上限矩阵 D:\keshe2\algorithm\out\tmp_arch\ceiling_v2\report.json。
背景（架构师 10-27）：
  - crnn_v2 已合入，复跑一致：难例开发集整牌 53.6%→68.1%，干得好。
  - 新的卡点在“找到了却读错”：你的调参集上 crnn_v2 读真值裁片 85.95%，读流水线候选裁片只有 70.60%；
    02 测得 db 上框命中的候选里，四角点平均偏差约 20 px（原图像素），“框对但角点偏 >12 px”占大多数。
    架构师用 ceiling.py 在 devhard_v1 同一批图上测了 crnn_v2：读真值四角点裁片——最暗档（db 亮度 ≤19）82.4%、tilt 94.0%、rotate 95.7%、db 68.3%；
    读实际定位候选裁片——58.8%、68.0%、83.7%、46.0%。
    也就是说读牌在“裁得准”时已经不错，在“裁得不准”时掉 20—26 个百分点。你第八批的顶点扰动是“小幅”，和真实误差不匹配。
任务：
  0. 调研（1 小时内，派子代理）：同类方案怎么让识别器容忍定位误差——LPRNet 的 LocNet / STN、LPTR-AFLNet 的 PTR 校正器与顶点扰动训练、
     用检测框外扩裁片直接识别的做法等。补充到 docs/research/01-整牌识别调研.md 新的一节，结尾写本批采用什么、不采用什么。
  1. 先量化真实误差：从 02 第九批的逐图 trace（或用流水线在 plates/v2 难例训练池上跑出的候选四角点）统计候选四角点相对真值的误差分布
     （按裁片宽高归一化，分 db、tilt 等子集），回执给分位数表。只能用训练池和你的调参集，不能用 devhard_v1、val1000_v1、测试清单、金样例。
  2. 训练 crnn_v3：在 crnn_v2 上续训（或按调研加一个小 STN 前端，二选一或都试，回执写理由）：
     顶点扰动按第 1 项的真实分布采样（不再是小幅）；提高难例流水线候选裁片的比例；增广保留第八批的光照/模糊。
     沿用第八批的调参集与选轮规则，devhard_v1 只在最后报告；置信度沿用温度校准。模型存 algorithm/models/crnn_v3.pt（≤5 MB），登记 manifest（只加你的条目）。
  3. 评测：
     a. 调参集：真值裁片与流水线候选裁片分别报（7 个子集 + 最暗档）。目标：候选裁片总体从 70.60% 提到 ≥82%，真值裁片不低于 crnn_v2。
     b. 整帧：LPR_CONFIG 临时配置（CNN v2 + corner_net_v1 + crnn_v3），val1000_v1 与 devhard_v1 上跑 first_ok 与 agree（agree 用默认阈值），与 crnn_v2 对比，
        单列 db 亮度 ≤19 一档与 tilt+rotate。val1000 整牌不得低于 crnn_v2 超过 0.5 个百分点（注意绿牌）。
     c. 单块 read() ≤10 ms（CPU 2 线程）；如加 STN，G1 整帧 P95 ≤200 ms。
  止损：训练 4 轮后候选裁片准确率比 crnn_v2 提升不到 3 个百分点，就停下，写清原因和调研里别人的做法，不要继续磨。
只允许新建或修改：algorithm/lpr/recognize/crnn.py、cnn_model.py、cnn.py、template.py、svm.py；algorithm/lpr_tools/train/；algorithm/lpr_tools/eval/ablation.py；
  algorithm/models/crnn_v3.pt 与 algorithm/models/manifest.json（只加你的条目）；algorithm/tests/ 下以 test_crnn、test_recognize、test_train 开头的文件；
  docs/research/01-整牌识别调研.md；D:\keshe2\algorithm\data\chars\ 与 D:\keshe2\algorithm\data\plates_synth\。
禁止：改 pipeline.py、config.py、recognize/base.py、default.toml；任何预训练权重、现成 OCR 或车牌识别库、复制他人代码、pip install；
      用测试清单、金样例、devhard_v1.csv、val1000_v1.csv、坏图名单里的图训练或调参；切换或合并分支；改其他工位。
协作方式：积极派子代理并行（调研、误差统计、训练、评测可并行，训练放后台，训练线程 ≤8）；看图先缩小到 ≤400 px，一次不超过 8 张。
汇报方式：完整汇报写到 algorithm/out/receipts/第九批.md；聊天里只回一行：回执 algorithm/out/receipts/第九批.md，commit <最后提交号>
```
