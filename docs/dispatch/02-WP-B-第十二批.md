# 派单 02 · 员工 B · 第十二批：plate_det_v1 接入准备（整帧时延、传统定位去留、无牌误检、尺度鲁棒性）+ 详细设计素材

```text
你是 M2 车牌识别模块执行者【员工 B，工位 02】。工作目录 D:\keshe2\workers\02，分支 feature/m2-wpb-locate。
推理与测试用你自己的 .venv；如需训练用 D:\keshe2\.venv-gpu。
必读：你第十一批回执；01 第十一批回执（box_reader_v2）；03 第十二批回执与 docs/research/03-三路仲裁集成规格.md；algorithm/reports/scoreboard.md 最上面两行。
背景（架构师 10-28）：
  - 你的 plate_det_v1 已验收合入 develop。架构师在 develop 复跑了你的 devhard 冻结评测，与回执一致。
    难例集 Top1 IoU≥0.7 99.5%（旧 corner_net_v1+传统融合 71%），四角点误差约 6 px，CPU 单次 detect P95 45 ms。
  - 回执里“直框 + crnn_v2 整牌 15.7%”是旧 CRNN 读未校正裁片，本来就不是它的输入，不代表检测器有问题；整牌效果由 03 用三路组合重算。
  - plate_det_v1 已能用现有入口加载（load_corner_detector 按架构分派），接入流水线只需把 runtime.corner_model_path 指向它，由架构师在 03 出新规格后切默认。
  - 下一步要面对本地实拍：停车场摄像头 1280×720 横幅、手机照片，车牌大小和 CCPD（720×1160 竖幅）不同；没车牌的画面也会进来。
任务：
  1. 整帧时延：在 out 下复制一份 default.toml，只把 runtime.corner_model_path 改成 models/plate_det_v1.pt，用它跑
     lpr_tools.eval.bench_latency（测试清单 320 张，只测时间、不看准确率，预热 5 次后 100 次，1280×720），报 P50/P95/最大值与各阶段耗时，和现默认（P95 92 ms）并列。
     另跑金样例（test_golden 同样口径，用这份临时配置）报几张对。不许改 default.toml。
  2. 传统定位去留：plate_det_v1 之后，pipeline._merge_candidates 仍会合并传统定位候选。在你的独立 hard holdout 2100 与 val 留出（不是固定名单）上比较
     “只用 plate_det_v1”与“plate_det_v1 + 传统定位合并”的 Top-1/Top-3 IoU≥0.7 召回、平均候选数、定位耗时；给出建议（保留 / 去掉 / 只在检测器无候选时兜底）。
  3. 无牌误检：NP 验证 601 张里 11.3% 出了候选。在 NP 验证与 hard holdout 上画“分数阈值 → 误检图率 / Top-1 召回”曲线，给推荐阈值；
     把阈值做成加载参数（默认值保持 0.25 不变），不改固定集上的任何东西。
  4. 尺度与画幅鲁棒性：在 hard holdout 上把原图按 0.4、0.6、0.8、1.0、1.3 倍缩放后放进 1280×720 画布（补边），报 Top-1 IoU≥0.7 召回与车牌像素宽度分布。
     若 0.6—1.3 倍任一档召回低于 95%：做 plate_det_v2——从 v1 继续训练，加多尺度/画幅随机缩放与补边增广，至少 10 个完整 epoch（全池），
     按 hard holdout 选模型，再报第 4 项全部档位与 devhard/val1000 冻结评测（只报告不调参）。v1 已够就不训，回执写明数字。
  5. 《详细设计》第 3 章素材（10-30 交，架构师整理）：新建 docs/design/ch3-定位检测器.md——结构（ResNet18 前三段 + FPN + 中心/框/四角点头，mermaid 结构图）、
     输入输出、损失、训练数据与排除规则、训练设置与轮数、开发集结果表、CPU 时延。书面中文，约 2 页，数字只引用已复核的。
资源与磁盘（用户机器，内存紧张）：你所有进程合计内存 ≤2.5 GB；GPU 与 01 共用，你最多 4 GB 显存；CPU 评测 2 线程、一次一个进程；
  新增磁盘 ≤2 GB，checkpoint 只留 best 与 last，不建新图片缓存；第 4 项若训练，单次训练有效墙钟 ≤5 小时，首个 epoch 后把每 epoch 耗时和推算总时长写进回执草稿。
  内存不足时照第十一批做法先存档再停，并在聊天里说明需要多少空闲内存。
只允许新建或修改：algorithm/lpr/corner_net.py、plate_det.py、locate.py；algorithm/lpr_tools/train_det.py、eval_det.py；
  algorithm/models/plate_det_v2.pt 与 algorithm/models/manifest.json（只加你的条目）；algorithm/tests/ 下以 test_det、test_corner、test_locate 开头的文件；
  docs/research/02-车牌定位调研.md；docs/design/ch3-定位检测器.md。
禁止：改 pipeline.py、config.py、boxcrop.py、recognize/、default.toml；车牌专用现成模型/库/权重、ultralytics/YOLO 库；复制他人代码；安装新包；
  用五份名单（testset、golden、devhard、val1000、undecodable）里的图训练或调参；切换或合并分支；改其他工位。
协作方式：积极派子代理并行；看图先缩小到 ≤400 px，一次不超过 8 张。
汇报方式：完整汇报写到 algorithm/out/receipts/第十二批.md；聊天里只回一行：回执 algorithm/out/receipts/第十二批.md，commit <最后提交号>
```
