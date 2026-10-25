# 派单 02 · 员工 B · 第七批：四角点网络（K1 Plan B-1）

```text
你是 M2 车牌识别模块执行者【员工 B，工位 02】。工作目录 D:\keshe2\workers\02，分支 feature/m2-wpb-locate，用 .venv\Scripts\python.exe。
必读：docs/M2-车牌识别-开发文档.md 第 2 章（C2：禁止任何预训练权重）、5.9 节最后两条（V0.6 神经网络接口）、14 章 K1；
      algorithm/lpr/recognize/base.py 的 CornerDetector；algorithm/lpr/corner_net.py（桩）；algorithm/lpr/pipeline.py 的 _merge_candidates。
背景：tonyswift868已决定提前启用 Plan B。流水线已接好四角点网络：网络候选排在传统定位候选前面，IoU≥0.5 去重。
      当前 develop：val 1000 整牌 56.3%，Top-3 定位召回 89.4%；难例开发集定位与识别都很差（夜间、倾斜、旋转最弱）。
任务：
  1. 数据：从 D:\keshe2\algorithm\data\index\ccpd_index.csv 的 train 划分取训练样本（含 ccpd_green train；ccpd_np 无牌图做负样本），
     val 划分做验证；一律排除测试清单与金样例。注意：CCPD 是 720×1160 竖图，而系统实际输入是 1280×720 横图，
     训练与推理都要按任意宽高比补边缩放（letterbox），并做平移、缩放、旋转、透视、亮度、模糊等增广。
  2. 模型：lpr/corner_net.py 实现 CornerNet（PyTorch 从零训练，不得使用任何预训练权重或现成检测模型），输出四个角点与“有车牌”置信度；
     可用“粗定位 → 裁剪 → 精修角点”两级结构。detect() 只返回有把握的候选（通常 0—2 个），source="net"，score 为置信度。
     训练脚本放 lpr_tools/train_corner.py，后台运行写日志，CPU 训练线程 ≤8（另外两个工位也在训练，别抢满 CPU）。
     模型存 algorithm/models/corner_net_v1.pt（≤5 MB，入库）并用 write_manifest_entry 登记。
  3. 评测（用 LPR_CONFIG 临时指向一个只设 runtime.corner_model_path 的配置，不改 default.toml）：
     角点平均误差（像素）；val 1000 与难例开发集 7×300 的 Top-3 定位召回和整牌准确率，与不开网络对比。
     目标：val Top-3 召回 ≥97%；难例开发集每个子集 Top-3 召回 ≥85%；val 整牌准确率比 56.3% 高；单帧 G1 P95 仍 ≤200 ms。
  4. rectify / rectify_candidates 对 source="net" 的候选直接利用网络角点（可再精修），在回执里说明做法。
只允许新建或修改：algorithm/lpr/corner_net.py、preprocess.py、locate.py、rectify.py、segment.py；algorithm/lpr_tools/train_corner.py、synth_ccpd.py、calib_layout.py、viz.py；
  algorithm/models/corner_net_v1.pt 与 algorithm/models/manifest.json（只增加你的条目）；algorithm/tests/ 下以 test_corner、test_preprocess、test_locate、test_rectify、test_segment 开头的文件；
  D:\keshe2\algorithm\data\calib\ 与 D:\keshe2\algorithm\data\corner\（训练数据缓存）。
禁止：改 pipeline.py、recognize/base.py、config 键（需要就在回执里提）；任何预训练权重、现成检测库（YOLO/ultralytics 等）、pip install；
      在测试清单或金样例上训练或调参；切换或合并分支；改其他工位。
协作方式：积极派子代理并行（如一个准备数据与增广、一个设计网络与训练、一个做评测与集成）；看图先缩小到 ≤400 px，一次不超过 8 张。
汇报方式：完整汇报写到 algorithm/out/receipts/第七批.md；聊天里只回一行：回执 algorithm/out/receipts/第七批.md，commit <最后提交号>
```
