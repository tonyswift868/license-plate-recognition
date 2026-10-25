# 派单 01 · 员工 A · 第五批：训练与推理对齐的字符样本 v1 和 CNN v1（A4）

```text
你是 M2 车牌识别模块执行者【员工 A，工位 01】。工作目录 D:\keshe2\workers\01，分支 feature/m2-wpa-model，用 .venv\Scripts\python.exe。
必读：algorithm/reports/scoreboard.md 第一行；D:\keshe2\algorithm\out\day1\val\report.json（逐图 failures）；
      docs/M2-车牌识别-开发文档.md 7.3 节（v1：改用 segment() 的真实分割结果，贴近推理分布）、8.2 节、12.5 节。
现状：CNN v0 在你的字符验证集上 99.65%，但真实流水线里逐位字符准确率只有 18.5%。原因之一：v0 字符按固定版式槽切图，
      而推理时字符图来自 segment() 收紧到墨迹的框，两者分布不一致（例：沪LE8887 识别成 沪LF8HH7）。02 号员工正在同时修分割。
任务：
  1. 先量化：在 val 上取“真值四角点校正 + 当前 segment() 分割成功且位数正确”的车牌，把 segment 输出的字符图喂给 CNN v0，
     统计逐位准确率；与 v0 自己的字符验证集对比，确认分布差多少。
  2. 字符样本 v1：build_chars 增加 --source segment 模式，用当前 develop 的 rectify/segment（真值四角点校正，可选再加定位候选校正）
     生成字符图（字符图必须是 segment 返回的 chars，即与推理完全同一路径），标签取 CCPD 真值；位数不对的整图丢弃并计数。
     输出 D:\keshe2\algorithm\data\chars\v1\，仍须 --exclude-manifest 测试清单，并剔除金样例（algorithm/tests/data/golden/golden_manifest.csv）。
  3. CNN v1：v1 分割字符为主，v0 固定槽字符与合成省份字为辅混合训练；在 v1 字符验证集上目标 ≥98%、省份字 ≥95%。
     模型存 algorithm/models/char_cnn_v1.pt（≤5 MB，入库）并登记 manifest.json。用 LPR_CONFIG 临时指向 v1，
     跑 eval_accuracy --split val --n 1000 --classifier cnn，给出整牌准确率与 v0 的对比（不改 default.toml）。
  4. 02 号员工的分割改进合入后，分割路径会变；在回执里写清 v1 样本用的是哪个 commit 的 segment，方便以后重建。
只允许新建或修改：algorithm/lpr/recognize/cnn_model.py、cnn.py、template.py、svm.py；algorithm/lpr_tools/train/；algorithm/lpr_tools/eval/ablation.py；
  algorithm/models/char_cnn_v1.pt 与 algorithm/models/manifest.json；algorithm/tests/ 下你名下的测试；docs/design/03-车牌识别.md；D:\keshe2\algorithm\data\chars\。
禁止：改 default.toml 与其他代码；用测试清单或金样例里的图训练或调参；pip install；切换或合并分支；改其他工位。
协作方式：积极派子代理并行（训练放后台，同时做量化分析）；看图先缩小到 ≤400 px，一次不超过 8 张。
汇报方式：完整汇报写到 algorithm/out/receipts/第五批.md；聊天里只回一行：回执 algorithm/out/receipts/第五批.md，commit <最后提交号>
```
