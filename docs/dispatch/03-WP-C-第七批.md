# 派单 03 · 员工 C · 第七批：CRNN 整牌训练数据集、评测支持神经网络

```text
你是 M2 车牌识别模块执行者【员工 C，工位 03】。工作目录 D:\keshe2\workers\03，分支 feature/m2-wpc-post-eval，用 .venv\Scripts\python.exe。
必读：docs/M2-车牌识别-开发文档.md 5.9 节最后两条（V0.6 神经网络接口）、7.3—7.5 节、14 章 K1；algorithm/lpr/recognize/base.py 的 PlateReader。
背景：tonyswift868已决定提前启用 Plan B。01 号员工做完 CNN v2 后要训练 CRNN 整牌识别（输入是校正后的整块车牌 252×80 / 276×80），你先把它的训练数据备好；
      02 号员工同时在训练四角点网络。
任务：
  1. 整牌数据集：lpr_tools/data/build_plates.py，从 ccpd_index.csv 的 train / val 划分生成整块车牌图与标签：
     (a) 用真值四角点透视校正（与 rectify 的输出尺寸一致：7 位 252×80、8 位 276×80）；
     (b) 再加“真实流水线候选”版本：用当前 develop 的 preprocess→locate→rectify_candidates 得到、且与真值框 IoU≥0.5 的校正图（贴近推理分布）；
     一律排除测试清单与金样例（路径与同字节副本都要排除），绿牌与各省份要有统计表，样本少的省份单列。
     输出 D:\keshe2\algorithm\data\plates\v1\{train,val}\ 与 labels.csv（列：file,plate_no,layout,color,source,orig_path），图片读写用 lpr.io_utils。
     规模：train 至少 15 万张、val 1 万张左右；放后台多进程跑，CPU 进程数 ≤6（另外两个工位在训练）。
  2. 评测支持：daily.py、eval_accuracy、bench_latency 都要能通过 --config（或 LPR_CONFIG）带上 runtime.corner_model_path / plate_reader_model_path 运行；
     报告里多列出“由 CRNN 兜底得到的结果数及其正确率”“由网络候选得到的结果数”（可从 RecognitionResult.stage_ms 是否含 reader / corner 与重放判断），
     便于架构师判断两个网络各自的贡献。
  3. 用当前 develop 跑一次 daily.py，产出一行积分板（不改 scoreboard.md）。
只允许新建或修改：algorithm/lpr_tools/data/、algorithm/lpr_tools/eval/（ablation.py 除外）、algorithm/testset/、
  algorithm/tests/ 下以 test_ccpd、test_manifest、test_eval、test_bench、test_sample、test_golden_pick、test_daily、test_build_plates 开头的文件；D:\keshe2\algorithm\data\plates\。
禁止：改运行时代码、配置、test_golden.py、scoreboard.md；在测试清单上调参；pip install；切换或合并分支；改其他工位。
协作方式：积极派子代理并行（数据集生成在后台跑时并行做评测支持）；看图先缩小到 ≤400 px，一次不超过 8 张。
汇报方式：完整汇报写到 algorithm/out/receipts/第七批.md；聊天里只回一行：回执 algorithm/out/receipts/第七批.md，commit <最后提交号>
```
