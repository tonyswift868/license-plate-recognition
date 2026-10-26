# 派单 01 · 员工 A · 第七批：CRNN 整牌识别（K1 Plan B-2）

```text
你是 M2 车牌识别模块执行者【员工 A，工位 01】。工作目录 D:\keshe2\workers\01，分支 feature/m2-wpa-model，用 .venv\Scripts\python.exe。
必读：docs/M2-车牌识别-开发文档.md 第 2 章（C2：禁止任何预训练权重）、5.9 节最后两条（V0.6 神经网络接口）、14 章 K1；
      algorithm/lpr/recognize/base.py 的 PlateReader；algorithm/lpr/recognize/crnn.py（桩）；algorithm/lpr/pipeline.py 的 _decode_text 与整牌兜底；
      algorithm/reports/scoreboard.md 最新一行（四角点网络已上线后的基线）；03 号员工的数据说明 D:\keshe2\algorithm\data\plates\v1\stats.md。
背景：tonyswift868已决定启用 Plan B。四角点网络已合入，验证集整牌准确率大幅提升；剩下的主要损失是分割失败与难例。
      流水线已接好 CRNN：某个四角点假设走分割路线没出 OK 时，把这块校正图交给 PlateReader.read() 整牌识别兜底。
任务：
  1. 数据：D:\keshe2\algorithm\data\plates\v1\（train 294,783 / val 10,000，labels.csv；含真值四角点校正图与真实流水线候选校正图）。
     省份极度偏斜（皖占绝大多数，20 多个省份 train < 1000），用 Pillow 字体合成整块车牌补足稀少省份（只在 lpr_tools 开发期使用，
     合成图只进训练集，不进验证集），并做类别均衡采样与增广（模糊、噪声、亮度、透视微扰、遮挡）。
  2. 模型：lpr/recognize/crnn.py 实现 CrnnReader（PyTorch 从零训练，CNN + 序列模型 + CTC，66 类 = 65 字符 + blank），
     read() 输入校正后的整块车牌 BGR（252×80 或 276×80），输出 (车牌字符串, 逐位置信度)；读不出返回 ("", [])。
     训练脚本 lpr_tools/train/train_crnn.py，后台运行写日志，CPU 训练线程 ≤8。模型存 algorithm/models/crnn_v1.pt（≤5 MB，入库），用 write_manifest_entry 登记。
  3. 评测：plates/v1 val 上整牌准确率（真值校正图、流水线校正图分开报），目标分别 ≥95%、≥90%；省份字单独报。
     再用 LPR_CONFIG 临时配置（runtime.model_path 用积分板最新一行的 CNN、corner_model_path = "models/corner_net_v1.pt"、
     plate_reader_model_path = "models/crnn_v1.pt"）跑 eval_accuracy --split val --n 1000 --seed 42 与 --dev-hard --n 300，
     与不开 CRNN 对比整牌准确率、自动通过率与自动通过精度（精度不得低于 99%）。不改 default.toml。
  4. 推理耗时：单块车牌 read() ≤10 ms（CPU 2 线程），G1 整帧 P95 仍 ≤200 ms。
只允许新建或修改：algorithm/lpr/recognize/crnn.py、cnn_model.py、cnn.py、template.py、svm.py；algorithm/lpr_tools/train/；algorithm/lpr_tools/eval/ablation.py；
  algorithm/models/crnn_v1.pt 与 algorithm/models/manifest.json（只加你的条目）；algorithm/tests/ 下以 test_crnn、test_recognize、test_train 开头的文件；D:\keshe2\algorithm\data\chars\ 与 D:\keshe2\algorithm\data\plates_synth\。
禁止：改 pipeline.py、recognize/base.py、default.toml；任何预训练权重、现成 OCR（EasyOCR/PaddleOCR 等）、pip install；
      用测试清单、金样例、D:\keshe2\algorithm\data\index\devhard_v1.csv 或 val1000_v1.csv 里的图训练；切换或合并分支；改其他工位。
协作方式：积极派子代理并行（合成数据、网络与训练、评测可并行，训练放后台）；看图先缩小到 ≤400 px，一次不超过 8 张。
汇报方式：完整汇报写到 algorithm/out/receipts/第七批.md；聊天里只回一行：回执 algorithm/out/receipts/第七批.md，commit <最后提交号>
```
