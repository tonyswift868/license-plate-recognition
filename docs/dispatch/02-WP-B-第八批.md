# 派单 02 · 员工 B · 第八批：难例定位（夜间 db、远距离 fn 等）

```text
你是 M2 车牌识别模块执行者【员工 B，工位 02】。工作目录 D:\keshe2\workers\02，分支 feature/m2-wpb-locate，用 .venv\Scripts\python.exe。
必读：algorithm/reports/scoreboard.md 最新一行；你第七批回执里的难例子集召回表；docs/M2-车牌识别-开发文档.md 第 2 章、7.4 节、12.5 节、14 章 K1。
背景：四角点网络已合入（val 整牌约 82%，以积分板为准），但难例子集里 ccpd_db（夜间）、ccpd_fn（远/近距离）召回仍 <85%，难例整牌准确率离 G2 的 85% 还远。
架构师决定（新）：难例子集可以用于训练，但必须排除以下四份名单中的图（路径与同字节副本都要排除）：
  D:\keshe2\algorithm\data\index\testset_manifest_v1.csv（冻结测试集）、algorithm/tests/data/golden/golden_manifest.csv（金样例）、
  D:\keshe2\algorithm\data\index\devhard_v1.csv（每日难例评测固定 2100 张）、D:\keshe2\algorithm\data\index\undecodable.csv（坏图）。
  剩下的 db/blur/tilt/rotate/fn/challenge/weather 图记为“难例训练池”，回执里报告各子集可用数量。
任务：
  1. 四角点网络 v2：在 train 划分基础上加入难例训练池（按子集均衡采样），重点补夜间、远距离、强倾斜；训练线程 ≤8，后台运行。
     模型存 algorithm/models/corner_net_v2.pt（≤5 MB，入库）并登记 manifest.json（只加你的条目）。
  2. 评测（LPR_CONFIG 临时把 corner_model_path 指向 v2，CNN 用积分板最新一行的版本）：难例开发集 7×300 每个子集的 Top-3 召回与整牌准确率、
     val 1000 整牌准确率，与 corner_net_v1 对比。目标：难例每个子集 Top-3 召回 ≥90%，val 整牌不下降，G1 P95 ≤200 ms。
  3. 若定位已够、但难例主要卡在校正或分割，回执里给出分阶段归因（定位/校正分割/识别各丢多少），供架构师决定下一步。
只允许新建或修改：algorithm/lpr/corner_net.py、preprocess.py、locate.py、rectify.py、segment.py；algorithm/lpr_tools/train_corner.py、synth_ccpd.py、calib_layout.py、viz.py；
  algorithm/models/corner_net_v2.pt 与 algorithm/models/manifest.json（只加你的条目）；algorithm/tests/ 下以 test_corner、test_preprocess、test_locate、test_rectify、test_segment 开头的文件；
  D:\keshe2\algorithm\data\corner\ 与 D:\keshe2\algorithm\data\calib\。
禁止：改 pipeline.py、recognize/base.py、default.toml；任何预训练权重或现成检测模型、pip install；用上面四份名单里的图训练或调参；切换或合并分支；改其他工位。
协作方式：积极派子代理并行；看图先缩小到 ≤400 px，一次不超过 8 张。
汇报方式：完整汇报写到 algorithm/out/receipts/第八批.md；聊天里只回一行：回执 algorithm/out/receipts/第八批.md，commit <最后提交号>
```
