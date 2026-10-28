# 派单 03 · 员工 C · 第十六批：检测器输入尺寸做成配置并在整条流水线上评 256/384/512（真实摄像头尺度），G3 视频预演

```text
你是 M2 车牌识别模块执行者【员工 C，工位 03】。工作目录 D:\keshe2\workers\03，分支 feature/m2-wpc-post-eval，用 .venv\Scripts\python.exe。
必读：你第十五批回执；02 第十二批回执 D:\keshe2\workers\02\algorithm\out\receipts\第十二批.md（尺度与分辨率实验、时延）；
      algorithm/lpr/plate_det.py、corner_net.py（02 第十二批已合入 develop 8f0e532）；algorithm/lpr_tools/eval_det.py 的尺度诊断；lpr_tools/eval/bench_video.py。
为什么现在做（架构师 10-28）：
  - 02 发现：把开发集图放进 1280×720 整帧后车牌变小，检测器输入 256 时 1.0 倍尺度召回只有约 81%，0.6 倍更低；输入 512 时各尺度都 ≥95%，
    但检测单次 P95 约 139 ms。G2 测的是原图，不受影响；11-03 和 M3 用真实摄像头联调时会受影响。
  - 检测器输入一改，三路读法和阈值都会变。01 的省份模型约 11 点交付，之后要重选一次阈值；输入尺寸必须在那之前定下来，否则阈值要选两次。
  - 本批只出数据和建议，不改默认值；最后由架构师和用户定。
任务：
  1. 配置：runtime 新增 detector_input_size（整数；0 = 用模型文件自带的输入尺寸，即现状 256；否则 128—1024 且为 32 的倍数），
     流水线加载 plate_det 时传入。default.toml 取 0，默认行为必须完全不变：用默认配置把 3100 张对第十三批期望逐张比，0 差异。
     test_contracts.py 的冻结键集合只加这一个键（架构师授权），其他断言不改。
  2. 原图口径（对应 G2）：detector_input_size = 384、512 各跑一次 val1000_v1 + devhard_v1 共 3100 张，和 256 的期望逐张对比，报：
     测试集组成折算的清晰/夜间/倾斜/污损、val 与 devhard 自动通过精度、改对/改错张数及代表样例。阈值沿用现默认，不重选。
  3. 摄像头口径：从 val1000_v1 与 devhard_v1 按子集分层抽 600 张（固定种子，清单入回执），按 02 的方法等比缩放到 0.6/0.8/1.0 倍后放进 1280×720 整帧，
     在 256/384/512 三种输入下跑整条流水线，报整牌正确率、自动通过精度、未检出（NO_PLATE）张数。
  4. 时延：256/384/512 各用 bench_latency 测一次 G1（与第十五批同条件），报总 P50/P95/最大值与 corner 阶段。
  5. G3 预演：用 val1000_v1 的图（不许用测试清单）合成一段 60 秒、1280×720、25 fps 的视频，按 bench_video 的 5 fps 抽帧跑默认配置，
     报端到端 fps（门槛 ≥15）。这只是工具与速度预演，正式 G3 要用真实停车场视频。视频文件 ≤100 MB，放 out 下。
  6. 结论表：每种尺寸一行，列原图四组、两个自动通过精度、摄像头三档、G1 P95；写明哪个尺寸在 G1 ≤300 ms 内同时满足原图不退步（下降 ≤1 个百分点）
     和摄像头 1.0 倍 ≥95%。不要写形容词，只给数。
资源与磁盘：不用 GPU；CPU 线程 ≤4（G1 固定 2 线程）；01 正在训练且全机内存紧张，你所有进程合计内存 ≤1 GB，评测一次只跑一个进程；
  新增磁盘 ≤500 MB（含预演视频）；pytest 用短 basetemp；临时目录不用清理，回执列出。
  预计：配置与测试 30 分钟；3100 张两次约 1.5 小时；摄像头 600 张 × 9 组约 1 小时；G1 三次约 15 分钟；G3 预演约 10 分钟。合计约 3—3.5 小时。
只允许新建或修改：algorithm/lpr/pipeline.py、algorithm/lpr/config.py（只为第 1 条）、algorithm/config/default.toml（只加这一个键）；
  algorithm/lpr_tools/eval/（ablation.py 除外）；algorithm/tests/ 下 test_contracts.py（只加这一个键）与 test_pipeline、test_config、test_eval、test_default 开头的测试；
  docs/design/ch3-仲裁与评测.md。
禁止：改默认取值以外的行为；改仲裁规则或阈值；改模型与 manifest、plate_det.py、corner_net.py、test_golden.py、scoreboard.md；在测试清单上跑准确率；
  pip install；切换或合并分支；改其他工位。
协作方式：可以派子代理并行写代码与测试，但同一时间只跑一个评测进程。
汇报方式：完整汇报写到 algorithm/out/receipts/第十六批.md；聊天里只回一行：回执 algorithm/out/receipts/第十六批.md，commit <最后提交号>
```
