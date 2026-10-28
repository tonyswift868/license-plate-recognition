# 派单 03 · 员工 C · 第十七批：换上省份模型 + 检测器 384，重选阈值，写进默认配置并逐张验收（测试集只准备，等架构师一句话再跑）

```text
你是 M2 车牌识别模块执行者【员工 C，工位 03】。工作目录 D:\keshe2\workers\03，分支 feature/m2-wpc-post-eval，用 .venv\Scripts\python.exe。
必读：你第十三、十六批回执与 thirteen_protocol.json；01 第十二批回执“续训与投入产出”及“非皖261与皖牌整牌”两节
      （D:\keshe2\workers\01\algorithm\out\receipts\第十二批.md）；algorithm/models/manifest.json 新增的 box_reader_v3、crnn_v4、char_cnn_v3 条目。
为什么做（架构师 10-28）：这是测试集正式评测前的最后一批。01 的省份模型已合入 develop（7089ee2），你第十六批的检测器尺寸实验也已合入（35c3551）。
  - 选型（架构师按数据定）：box_reader_v3、crnn_v4 采用；char_cnn_v3 不采用，继续用 char_cnn_v2。
    原因：char_cnn_v3 完整分割路线的真实外省留出从 93.42% 降到 92.59%，非皖 261 张读成“皖”从 9 张增到 15 张。
  - 检测器输入用 384：摄像头 1.0 倍整牌 97.0%，原图相对 256 最多降 0.2 个百分点，G1 P95 86.8 ms。
    512 在 0.6 倍更好，但绿牌降到 94.6%、P95 204.5 ms。
  - 模型和检测尺寸一换，三路读法全变，所以四个门槛都要重选。规则族 011 + T20_A 不变。
任务：
  1. 候选 C1 = char_cnn_v2 + crnn_v4 + box_reader_v3 + plate_det_v1（detector_input_size=384）：
     用 three_route_workflow 在 val1000_v1 + devhard_v1 上生成新 trace。按 thirteen_protocol.json 原样的 fit/hold 分半与 exact search 重选
     three_route_agree_min_confidence、three_route_box_solo_min_confidence。
     S/R 单路门槛原值来自第十批旧模型，不适用于 crnn_v4，solo_min_confidence_segment / reader 也按同一 fit 口径重选，用同一套选法。
     选法写定后再看结果。附加约束：val1000 与 devhard 全集自动通过精度都 ≥99.5%。fit 选出的值不满足时，只允许沿 exact search 的下一档收紧，
     直到两集合全集都满足；回执写明收紧了几档、少了多少张自动通过。
  2. 退路：C1 收紧后仍达不到，或四组折算任一组低于门槛（清晰 95、其余 85），就同法评 C2（把 box_reader_v3 换回 box_reader_v2，其余同 C1）；C1 达标时不评 C2。
  3. 选中的组合写进 default.toml，只改模型路径、detector_input_size 和四个阈值，键不增删。
     当前默认另存为 config/three_route_v2_det256.toml 留档，只写不同的值。生成新的逐张期望文件（val1000 + devhard 共 3100 张，含 route、候选和 rect 索引），
     再用 PlateRecognizer.from_config()（不传路径）真实跑 3100 张逐张对比，必须 0 差异。
  4. 报告：
     a. 新旧默认（当前默认 vs 新默认）在 3100 上的四组折算、两个自动通过精度、改对/改错张数；
     b. 系统级省份投入产出：01 的真实省份留出（province_holdout.csv）与合成 31 省面板，用新旧默认整条流水线各跑一次，
        报整牌正确率、省份字正确率、读成“皖”张数、自动通过精度；
     c. 摄像头 600 张（第十六批同一清单）0.6/0.8/1.0 三档整牌与自动通过精度；
     d. G1（同第十六批条件）、默认配置下 test_golden 张数、全量 pytest 一次、G7 覆盖率。
  5. 测试集：frozen_once 的准备命令按新默认重跑一次（不带正式 flag），回执末尾写好唯一正式命令。
     正式命令本批不许跑，等架构师复核后单独发话。
资源与磁盘：不用 GPU；CPU 线程 ≤4（G1 固定 2 线程）；你所有进程合计内存 ≤1.5 GB，评测一次只跑一个进程；新增磁盘 ≤800 MB；临时目录不用清理，回执列出。
  预计：C1 trace 与重选约 1—1.5 小时；默认验收 3100 张约 45 分钟；省份面板两套新旧各一次约 40 分钟；摄像头约 30 分钟；
  pytest、G1、G7 约 20 分钟。合计约 3.5—4 小时；若要评 C2，再加约 1.5 小时。
只允许新建或修改：algorithm/config/default.toml（只改上列取值）、algorithm/config/three_route_v2_det256.toml（新建）；algorithm/lpr_tools/eval/（ablation.py 除外）；
  algorithm/tests/ 下 test_default、test_eval、test_pipeline、test_config 开头的测试（只为新默认值更新期望，断言不放宽）；docs/design/ch3-仲裁与评测.md；docs/design/03-车牌识别.md（只填“【待填】”占位与新默认数字）。
禁止：改仲裁规则族、pipeline.py、config.py、arbitration.py 逻辑；改模型与 manifest、test_golden.py、test_contracts.py、scoreboard.md；
  在测试清单上跑准确率；pip install；切换或合并分支；改其他工位。
协作方式：可以派子代理并行写工具与测试，但同一时间只跑一个评测进程。
汇报方式：完整汇报写到 algorithm/out/receipts/第十七批.md；聊天里只回一行：回执 algorithm/out/receipts/第十七批.md，commit <最后提交号>
```
