# 派单 02 · 员工 B · 第十一批：像样的车牌检测器（预训练骨干，框 + 四角点），用 GPU 训足

```text
你是 M2 车牌识别模块执行者【员工 B，工位 02】。工作目录 D:\keshe2\workers\02，分支 feature/m2-wpb-locate。
必读：docs/M2-架构师交接.md 第 1 节与第 6 节 0''' 条（10-27 更正）；你第七至十批回执与 docs/research/02-车牌定位调研.md；
      algorithm/lpr/boxcrop.py（crop_box）；algorithm/lpr/recognize/base.py 的 CornerDetector；架构师 ceiling_v2 报告 D:\keshe2\algorithm\out\tmp_arch\ceiling_v2\report.json；
      参考论文 https://pmc.ncbi.nlm.nih.gov/articles/PMC11086086/（YOLOv5，640×640，训练 300 轮；CCPD 检测 DB 97.7%、Tilt 98.9%）。
背景（架构师 10-27，用户已拍板）：
  - 定位做不好不是方法不行，是网络太小、训得太少：corner_net_v1 第一级只有 9,813 个参数，只在 12,000 张清晰图上训了约 2 轮，一张难例都没见过；
    v3、精修器各只有 3—6 万次样本曝光。参考的 YOLOv5 有数百万参数，10 万张图训 300 轮。之前“训两轮不涨就停”是架构师定错了，作废。
  - 新路线把读牌改成“检测框外扩后直接读”（01 本批），所以检测器第一要务是框准、框全（把整块车牌框进去），四角点排第二（给旧的校正路线用）。
  - 用户新决定：可以用 GPU 训练，推理仍在 CPU；可以用 torchvision 自带的 ImageNet 预训练骨干；模型 ≤20 MB；车牌专用现成模型和 YOLO/ultralytics 库仍不能用。
任务：
  0. 调研（1 小时内，派子代理）：YOLOv5/YOLOv8-pose、CenterNet、RetinaFace 这类“框 + 关键点”检测器在 CCPD 上的结构、输入尺寸、增广（马赛克、亮度、透视）、训练轮数，
     以及 CPU 推理代价。补到 docs/research/02-车牌定位调研.md 新一节，结尾写采用什么结构、为什么。
  1. 检测器 plate_det_v1：自己写（结构可参照 CenterNet 或 YOLO，骨干用 torchvision 预训练的 MobileNetV3、ResNet18 等），输出车牌框、四角点、置信度，
     实现 CornerDetector 协议（PlateCandidate 的 bbox 与 corners 都要填）。在 corner_net.py 现有的加载入口里按 checkpoint 的 architecture 分派（同第九批 KeypointNet 的做法），
     这样 runtime.corner_model_path 指向新模型即可加载。模型 ≤20 MB；CPU 推理（2 线程，1280×720 补边口径）单次 ≤60 ms。
  2. 数据与训练（GPU）：训练池 = ccpd_index 里 split=train 的 base 与 green，加难例训练池，加 NP 负例；排除五份名单：testset_manifest_v1、golden_manifest、devhard_v1、val1000_v1、undecodable（路径与同字节）。
     你的独立 holdout 继续不进训练，只用来选模型。读图可用 cv2.IMREAD_REDUCED_COLOR_2 半分辨率解码或 torchvision.io.decode_jpeg 在 GPU 上解码，不另存缩小图。
     用共享 GPU 环境 D:\keshe2\.venv-gpu\Scripts\python.exe（PYTHONPATH=algorithm，在你的工作目录运行），--device cuda。训练量至少 40 个完整 epoch（每个 epoch 过完全部训练池）。
     训练脚本也要能 --device cpu 跑通一次小冒烟；推理与测试一律用你自己的 .venv（CPU）。
  3. 评测（devhard_v1、val1000_v1 只报告，不调参）：
     a. 框：各子集 Top-1、Top-3 召回（IoU ≥0.5 与 ≥0.7），单列最暗档（文件名亮度 ≤19）。目标：devhard 各子集 Top-1 IoU≥0.7 ≥95%，val ≥99%。
     b. 用你的框 + crop_box（pad_x=0.15、pad_y=0.25）+ crnn_v2 离线读一遍，报各子集整牌准确率（只作参考，01 的 box_reader 交回后由 03 复测）。
     c. 框命中候选的四角点平均误差，与 corner_net_v1 对比。
     d. NP 601 张验证图上的误检数。
  止损：训满第 2 项的训练量后，holdout 上 IoU≥0.7 召回仍不比 corner_net_v1 + 传统定位融合高，才停下写原因。
资源与磁盘（这是用户的机器，内存紧张，必须遵守）：
  - GPU 与 01 共用（RTX 5070 Ti，12 GB）：你最多占 5 GB 显存（torch.cuda.set_per_process_memory_fraction(0.42)）。
  - 内存：你所有进程合计 ≤2.5 GB（DataLoader worker ≤2，torch 线程 ≤4，不要把图片整批读进内存）；开工前看空闲内存，低于 4 GB 就先等或再减 worker。
  - 磁盘：不生成新的图片数据集；checkpoint 只留 best 与 last；评测只存数字和 ≤8 张缩略图；pytest 临时目录跑完就删。本批新增磁盘合计 ≤3 GB，回执里报实际占用。
  - 单次训练墙钟目标 ≤8 小时；预计超出时先缩小规模并在回执写明，不要无限跑。
只允许新建或修改：algorithm/lpr/corner_net.py、preprocess.py、locate.py、rectify.py、segment.py，新建 algorithm/lpr/plate_det.py；
  algorithm/lpr_tools/train_corner.py、synth_ccpd.py、calib_layout.py、viz.py、train_refine.py、eval_refine.py，新建 algorithm/lpr_tools/train_det.py、algorithm/lpr_tools/eval_det.py；
  algorithm/models/plate_det_v1.pt 与 algorithm/models/manifest.json（只加你的条目）；
  algorithm/tests/ 下以 test_corner、test_refine、test_det、test_preprocess、test_locate、test_rectify、test_segment 开头的文件；
  docs/research/02-车牌定位调研.md；D:\keshe2\algorithm\data\corner\ 与 D:\keshe2\algorithm\data\calib\。
禁止：改 pipeline.py、config.py、boxcrop.py、recognize/、default.toml；车牌专用现成模型/库/权重、ultralytics/YOLO 库；复制他人代码；安装 .venv-gpu 与 .venv 之外的新包；
      用五份名单里的图训练或调参；切换或合并分支；改其他工位。
协作方式：积极派子代理并行（调研、数据、训练、评测可并行，训练放后台）；看图先缩小到 ≤400 px，一次不超过 8 张。
汇报方式：完整汇报写到 algorithm/out/receipts/第十一批.md；聊天里只回一行：回执 algorithm/out/receipts/第十一批.md，commit <最后提交号>
```
