# 派单 01 · 员工 A · 第十一批：box_reader_v2——补容量、补倾斜、补绿牌和夜间样本，训足

```text
你是 M2 车牌识别模块执行者【员工 A，工位 01】。工作目录 D:\keshe2\workers\01，分支 feature/m2-wpa-model。
必读：你第十批回执；架构师三路组合评测 D:\keshe2\algorithm\out\tmp_arch\three_v1\report.json（及同目录 live.jsonl.gz）；03 第十一批回执；docs/research/01-整牌识别调研.md。
背景（架构师 10-27）：
  - box_reader_v1 做对了一件大事：实际候选框上最暗档 58.8%→74.5%、db +14、blur +17.7、challenge +9.3，和校正路线正好互补。
    架构师已复跑你的实际候选框数字，与回执逐张一致。三路组合（分割+CNN、校正 CRNN、框直接读）在两个开发集自动通过精度 ≥99.5% 的前提下，
    未自动通过时按“同块两票一致优先、否则取框读法”回退，最暗档已到 76.5%、难例总体 74.6%。
  - 但 v1 有三处明显短板，正是这批要补的：
    ① 容量太小：187k 参数、read 平均只用 2.2 ms，而预算是 10 ms——你为 CPU 预算把骨干截得太狠了；
    ② 倾斜读不好：tilt 56%、rotate 71%（校正路线 68%、84%），水平序列 CTC 对斜着的整串字符不友好；
    ③ 样本失衡：绿牌（8 位）只占训练 2.7%，实际绿牌 71.2%（校正路线 90.4%）；db 只占 1.8%。
任务：
  0. 调研（1 小时内，补到调研文档新一节）：识别器怎样读倾斜 30—45° 的整串字符而不依赖外部校正——STN/LocNet 前端、二维注意力或并行解码（PDLPR、SAR 一类）、
     更高输入分辨率；以及多类别不均衡下的分组均衡采样。结尾写本批采用什么。
  1. box_reader_v2：在 v1 基础上——
     a. 加大容量，用足 CPU 预算：单次 read() 平均 ≤6 ms、P95 ≤8 ms（2 线程），模型 ≤20 MB；可用完整 MobileNetV3 或 ResNet18 前几层等 ImageNet 预训练骨干；
     b. 针对倾斜：按调研加 STN 前端或二维注意力 / 并行解码（至少实现一种，回执写理由），可适当提高输入分辨率；
     c. 分组均衡采样：绿牌（8 位）、db、tilt、rotate 等子集按组均衡，不再按原图比例；扰动框分布沿用你第十批的实测分布；
     d. 训练量：至少 60 个完整 epoch（可复用你第十批的裁片缓存 D:\keshe2\algorithm\data\boxcrop_cache\box_reader_v1，外扩 50% 足够覆盖抖动）；按调参集选模型，温度校准同前。
     模型存 algorithm/models/box_reader_v2.pt 并登记 manifest；工厂 load_box_reader 能按 checkpoint 架构加载 v1、v2。
  2. 评测（只报告，不调参）：
     a. 调参集真值框与实测扰动框，分子集 + 最暗档 + tilt+rotate + 8 位牌单列。目标：真值框 tilt ≥90%、最暗档 ≥90%、总体 ≥ crnn_v2 真值校正裁片的 85.95%。
     b. 实际候选框（corner_net_v1 + 传统定位 Top-3，取置信度最高的合法读法）：val1000_v1、devhard_v1 各子集。目标：绿牌 ≥90%、最暗档 ≥80%、tilt+rotate ≥75%。
     c. 用 03 的工具一条命令复测三路组合（把 --box-model 换成 v2），报最暗档、tilt+rotate、绿牌与两个集合的自动通过精度：
        .venv\Scripts\python.exe -m lpr_tools.eval.three_route --fresh --search --config algorithm/config/default.toml --box-model algorithm/models/box_reader_v2.pt --box-loader lpr.recognize:load_box_reader --box-pad-x <v2 的 pad_x> --box-pad-y <v2 的 pad_y> --live-output algorithm/out/receipts/batch11/three-route-v2.jsonl.gz --out algorithm/out/receipts/batch11/three-route-v2.json
  止损：训满第 1 项 d 的训练量后，若调参集真值框 tilt 仍低于 80% 且实际候选框各目标都没提升，才停下写原因。
资源与磁盘（用户机器，内存紧张）：GPU 与 02 共用，你最多 5 GB 显存；你所有进程合计内存 ≤3 GB；复用已有缓存，不再新建图片缓存或数据集；checkpoint 只留 best 与 last；
  本批新增磁盘 ≤1 GB；单次训练墙钟目标 ≤6 小时，第 1 个 epoch 后把每 epoch 耗时和推算总时长写进回执草稿。训练用 D:\keshe2\.venv-gpu，推理与测试用你自己的 .venv。
只允许新建或修改：algorithm/lpr/recognize/（base.py 除外）；algorithm/lpr_tools/train/；algorithm/lpr_tools/eval/ablation.py；
  algorithm/models/box_reader_v2.pt 与 algorithm/models/manifest.json（只加你的条目）；algorithm/tests/ 下以 test_crnn、test_recognize、test_train、test_box_reader 开头的文件；docs/research/01-整牌识别调研.md。
禁止：改 pipeline.py、config.py、boxcrop.py、recognize/base.py、default.toml；车牌专用现成模型/库/权重；复制他人代码；安装新包；用五份名单里的图训练或调参；切换或合并分支；改其他工位。
协作方式：积极派子代理并行；看图先缩小到 ≤400 px，一次不超过 8 张。
汇报方式：完整汇报写到 algorithm/out/receipts/第十一批.md；聊天里只回一行：回执 algorithm/out/receipts/第十一批.md，commit <最后提交号>
```
