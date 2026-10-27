# 派单 03 · 员工 C · 第十四批：把三路仲裁 + plate_det_v1 接进流水线（实现、3100 张逐张验收、G1 时延），备好测试集一次性评测

```text
你是 M2 车牌识别模块执行者【员工 C，工位 03】。工作目录 D:\keshe2\workers\03，分支 feature/m2-wpc-post-eval，用 .venv\Scripts\python.exe。
必读：你第十三批回执；docs/research/03-三路仲裁集成规格.md 第 7 节（v2：011 + T20_A）与第 3、4 节（A/M/L 定义、业务状态投影）；
      algorithm/lpr/pipeline.py、config.py、config/default.toml；algorithm/lpr/recognize/ 下 load_box_reader、load_corner_detector；algorithm/lpr/boxcrop.py。
背景（架构师 10-28）：
  - 你第十三批已合入 develop（c3a4d4b），架构师用 main/expected-runtime.jsonl.gz 对官方索引逐张复算，与回执一致。
  - 开发集折算四项 G2 都过线，但三路仲裁还没接进流水线，所有数字都是离线重放，不算数。本批由你把它接进流水线。
    pipeline.py、config.py 原归架构师，本批授权你改；架构师只验收。
  - 01 正在训省份模型（box_reader_v3、crnn_v4、char_cnn_v3），上午交回执，届时只换配置里的模型路径和阈值，所以阈值、模型路径都必须走配置，不许写死在代码里。
任务：
  1. 实现（按规格 v2 第 7 节逐字实现 011 + T20_A，不改规则、不加规则）：
     a. 配置：runtime 新增 box_reader_model_path（空 = 不启用）；postprocess.arbitration 增加 three_route；新增 three_route_agree_min_confidence、
        three_route_box_solo_min_confidence、three_route_legacy_sr_agreement、three_route_segment_solo、three_route_reader_solo、
        three_route_province_support_min_routes、fallback_predicted_tilt_min_deg；nonok_fallback 增加 predicted_tilt_then_corrected。
        S/R 单路门槛复用既有 solo_min_confidence_segment / solo_min_confidence_reader，旧 agree_min_confidence 只给 L 用。
     b. default.toml 只补新键，取值保持当前默认行为完全不变（box_reader_model_path = ""、arbitration 仍为 agree）。
        新建 algorithm/config/three_route_v2.toml，只写与默认不同的键：corner_model_path = models/plate_det_v1.pt、box_reader_model_path = models/box_reader_v2.pt、
        arbitration = three_route、规格 7.2 的完整浮点阈值、nonok_fallback = predicted_tilt_then_corrected、fallback_predicted_tilt_min_deg = 20.0。
     c. 流水线：自动通过一出现就返回（早返回，G1 按这个测）；B 每个候选最多读一次并在该候选的各 rect 间复用，用官方 crop_box，pad 从权重读；
        仲裁决策写成不依赖图像的纯函数（可放新文件 algorithm/lpr/arbitration.py），输入为各候选各 rect 的 S/R/B 读法，便于单测与以后换阈值。
        PlateRecognizer 对外接口、recognize 永不抛异常、各状态码语义不变；version 字符串追加框读模型版本。
  2. 逐张验收（本批的硬指标）：用 PlateRecognizer.from_config("config/three_route_v2.toml") 按固定顺序跑 val1000_v1 + devhard_v1 共 3100 张，
     与 algorithm/out/receipts/第十三批/main/expected-runtime.jsonl.gz 比：status、plate、route 精确相等，confidence 绝对误差 ≤1e-7；
     能拿到候选/rect 索引就一并比。必须 0 差异；有差异改实现，不许改期望文件或用容差改变 >= 比较。差异样本逐张写进回执直到清零。
     另：默认配置不变——用默认配置在改动前后各跑 val1000_v1，逐张 status/plate/confidence 一致；金样例 test_golden 通过；全量 pytest 通过。
  3. G1 时延：用 three_route_v2.toml 跑 lpr_tools.eval.bench_latency（测试清单 320 张、1280×720、warmup 5、n=100、2 线程、不读真值），
     报总 P50/P95/最大值和各阶段（含框读）P50/P95；同条件再测一次默认配置作对照。在同一配置上跑金样例 20 张，报几张对。
  4. 测试集一次性评测（本批只准备，不运行）：写好在冻结测试清单上跑 G2 的一条命令与汇总脚本（按清晰/夜间/倾斜/污损四组和自动通过精度出表），
     用 2—3 张非测试集图片做冒烟。等 01 的省份模型换进配置、架构师通知后才在测试集上跑，且只跑一次。
  5. 单测：仲裁纯函数覆盖 S/R、S/B、R/B 配对顺序、所有 >= 边界、配对输出左路线原置信度、同 rect 双 solo 先 S、B-solo 在该候选全部 rect 之后、
     T20_A 角度（候选 0 预测四角、缺失或非有限值走 M）、白黑牌 UNSUPPORTED、INVALID_FORMAT 保留与 SEG_FAILED 映射；配置新键的校验。
  6. 在 docs/design/ch3-仲裁与评测.md 补一小节“流水线实现”：调用顺序、早返回、新配置键表、逐张验收与 G1 结果（只引用本批实测数）。
资源与磁盘：不用 GPU；CPU 线程 ≤4（G1 固定 2 线程）；你所有进程合计内存 ≤1.5 GB，评测一次只跑一个进程；新增磁盘 ≤300 MB；pytest 用短 basetemp。
  预计：实现与单测 1—2 小时；3100 张逐张跑约 40—60 分钟；默认配置 val1000 前后两次各约 10 分钟；G1 约 5 分钟。
只允许新建或修改：algorithm/lpr/pipeline.py、algorithm/lpr/config.py、algorithm/lpr/arbitration.py（新建）、algorithm/config/default.toml（只补新键）、
  algorithm/config/three_route_v2.toml（新建）；algorithm/lpr_tools/eval/（ablation.py 除外）；algorithm/tests/ 下以 test_pipeline、test_config、test_arbitration、
  test_three_route、test_eval 开头的文件；docs/research/03-三路仲裁集成规格.md；docs/design/ch3-仲裁与评测.md。
禁止：改模型与 manifest、recognize/base.py、boxcrop.py、corner_net.py、plate_det.py、test_golden.py、scoreboard.md；把默认配置切到 three_route；
  本批在测试清单上跑准确率；改仲裁规则或阈值；pip install；切换或合并分支；改其他工位。
协作方式：积极派子代理并行写实现与单测，但同一时间只跑一个评测进程。
汇报方式：完整汇报写到 algorithm/out/receipts/第十四批.md；聊天里只回一行：回执 algorithm/out/receipts/第十四批.md，commit <最后提交号>
```
