# 派单 01 · 员工 A · 第十批：“检测框直接读”的整牌识别（不做透视校正），用 GPU 训足

```text
你是 M2 车牌识别模块执行者【员工 A，工位 01】。工作目录 D:\keshe2\workers\01，分支 feature/m2-wpa-model。
必读：docs/M2-架构师交接.md 第 1 节与第 6 节 0''' 条（10-27 更正）；你第八、九批回执与 docs/research/01-整牌识别调研.md；
      algorithm/lpr/boxcrop.py（crop_box，新契约）与 algorithm/lpr/recognize/base.py 的 BoxReader；03 第十批回执的 crop_gap 结论；
      参考论文 https://pmc.ncbi.nlm.nih.gov/articles/PMC11086086/（YOLOv5 检测 + PDLPR 识别，不做校正，CCPD DB 99.5%、Tilt 99.3%）。
背景（架构师 10-27，用户已拍板）：
  - 之前“难例做不到”的结论是错的。参考系统不做透视校正，检测框裁下来直接读，DB、Tilt 都到 99%。
    我们的主流程依赖四角点精确校正，角点平均偏约 20 px 就裁坏：crnn_v2 读真值校正裁片 tilt 94%，读实际裁片只有 68%。
  - 识别模型也训得太少：crnn_v2 只过了约 1.3 遍数据，参考识别器训了 1000 轮。之前派单里“训两轮不涨就停”的止损规则作废，这批先训够再判断。
  - 用户新决定：① 可以用 GPU 训练，推理仍在 CPU；② 可以用 torchvision 自带的 ImageNet 预训练骨干（MobileNet、ResNet18 等），
    车牌专用的现成模型、库、权重仍不能用；③ 模型文件上限放宽到 20 MB。
任务：
  0. 调研（1 小时内，派子代理）：PDLPR、LPRNet、CRNN 类识别器怎样直接读“检测框裁片”（带背景、倾斜、不校正）——输入尺寸与颜色、外扩比例、
     框抖动增广、训练轮数与学习率、怎样读倾斜 30—45° 的车牌（STN、注意力，还是只靠数据）。补到 docs/research/01-整牌识别调研.md 新一节，结尾写本批采用什么。
  1. 数据：训练池 = ccpd_index 里 split=train 的 base 与 green，加难例训练池（db、blur、tilt、rotate、fn、challenge、weather），
     排除五份名单：testset_manifest_v1、golden_manifest、devhard_v1、val1000_v1、undecodable（路径与同字节都排除）；你第八批的难例调参集继续不进训练。
     样本从原图现场生成：真值四角点的外接框 → 按实测候选误差随机抖动（位移、缩放、宽高比，参考你第九批的误差统计）→ crop_box。
     读图提速二选一：用 torchvision.io.decode_jpeg 在 GPU 上解码；或生成一份临时裁片缓存（真值框外扩 50% 的原分辨率区域，JPEG，≤3 GB），训练结束后在回执里列出路径以便清理。
     可选：在训练池里抽约 2 万张，用 corner_net_v1 + 传统定位跑出真实候选框（只存框坐标 CSV），混入训练。
  2. 模型 box_reader_v1：结构按调研定，可用 ImageNet 预训练骨干。输入是 crop_box 裁片，外扩比例 box_pad 存进模型文件。
     在 lpr/recognize/ 下新文件实现 BoxReader（from_file、工厂函数），CPU 单次 read() ≤10 ms（2 线程），模型文件 ≤20 MB。
     存 algorithm/models/box_reader_v1.pt 并登记 manifest；置信度做温度校准（沿用你第八批的做法）。
  3. 训练（GPU）：用共享 GPU 环境 D:\keshe2\.venv-gpu\Scripts\python.exe（PYTHONPATH=algorithm，在你的工作目录运行），--device cuda。
     训练量至少 30 个完整 epoch（每个 epoch 过完全部训练池），按调参集选模型。训练脚本也要能 --device cpu 跑通一次小冒烟；推理与测试一律用你自己的 .venv（CPU）。
  4. 评测（都只报告，不用来调参）：
     a. 调参集：真值外接框（不抖动）与按实测误差抖动的框，分子集报，另单列最暗档（文件名亮度 ≤19）和 tilt+rotate。目标：真值框下各难例子集 ≥95%，最暗档 ≥90%，tilt ≥95%。
     b. 实际候选框：在 val1000_v1、devhard_v1 上，取 corner_net_v1 + 传统定位的 Top-3 候选框，逐个用 box_reader 读，取置信度最高的合法读法，报各子集整牌准确率（离线脚本，不改 pipeline.py）。
        与架构师的 ceiling_v2 报告（crnn_v2 读校正裁片）对比。
  止损：训满第 3 项的训练量后，若调参集真值框整牌准确率仍不比 crnn_v2 读真值校正裁片高，才停下写原因。
资源与磁盘（这是用户的机器，内存紧张，必须遵守）：
  - GPU 与 02 共用（RTX 5070 Ti，12 GB）：你最多占 5 GB 显存（torch.cuda.set_per_process_memory_fraction(0.42)）。
  - 内存：你所有进程合计 ≤2.5 GB（DataLoader worker ≤2，torch 线程 ≤4，不要把图片整批读进内存）；开工前看空闲内存，低于 4 GB 就先等或再减 worker。
  - 磁盘：除上面那份临时裁片缓存外，不生成新的图片数据集；checkpoint 只留 best 与 last；评测只存数字和 ≤8 张缩略图；pytest 临时目录跑完就删。本批新增磁盘合计 ≤4 GB，回执里报实际占用。
  - 单次训练墙钟目标 ≤8 小时；预计超出时先缩小规模并在回执写明，不要无限跑。
只允许新建或修改：algorithm/lpr/recognize/（base.py 除外）；algorithm/lpr_tools/train/；algorithm/lpr_tools/eval/ablation.py；
  algorithm/models/box_reader_v1.pt 与 algorithm/models/manifest.json（只加你的条目）；algorithm/tests/ 下以 test_crnn、test_recognize、test_train、test_box_reader 开头的文件；
  docs/research/01-整牌识别调研.md；D:\keshe2\algorithm\data\chars\、D:\keshe2\algorithm\data\plates_synth\ 与 D:\keshe2\algorithm\data\boxcrop_cache\（临时缓存）。
禁止：改 pipeline.py、config.py、boxcrop.py、recognize/base.py、default.toml；车牌专用的现成模型、库或权重（HyperLPR、LPRNet 公开权重、EasyOCR、PaddleOCR、ultralytics 等）；
      复制他人代码；安装 .venv-gpu 与 .venv 之外的新包；用五份名单里的图训练或调参；切换或合并分支；改其他工位。
协作方式：积极派子代理并行（调研、数据、训练、评测可并行，训练放后台）；看图先缩小到 ≤400 px，一次不超过 8 张。
汇报方式：完整汇报写到 algorithm/out/receipts/第十批.md；聊天里只回一行：回执 algorithm/out/receipts/第十批.md，commit <最后提交号>
```
