# 派单 01 · 员工 A · 第六批：按新分割重建字符、CNN v2、易混字专项

```text
你是 M2 车牌识别模块执行者【员工 A，工位 01】。工作目录 D:\keshe2\workers\01，分支 feature/m2-wpa-model，用 .venv\Scripts\python.exe。
必读：algorithm/reports/scoreboard.md 前两行；03 号员工第五批回执里的混淆 Top20（D:\keshe2\workers\03\algorithm\out\receipts\第五批.md）。
现状（develop，cnn v1）：val 1000 整牌 44.6%，cnn v0 同条件 31.1%。02 号员工第五批把送进 CNN 的字符图改成“全高槽位上下文”，
      而 v1 训练字符是改之前的 segment 生成的，两者仍有错配。高频混淆：0→1、皖→沪、D→1、8→2、7→1、0→L、A→D、皖→渝、E→F。
任务：
  1. 字符 v2：用当前 develop 的 segment（全高槽位上下文）重建字符，--source segment，真值四角点校正为主，另加“定位候选校正后与真值框 IoU≥0.5”的样本
     （贴近真实推理）；仍须排除测试清单与金样例。输出 D:\keshe2\algorithm\data\chars\v2\，回执写清所用 develop commit。
  2. CNN v2：v2 字符为主训练，易混字专项（按上面混淆对加权采样或难例挖掘，可加针对性增广），省份字继续用合成补少数类。
     模型存 algorithm/models/char_cnn_v2.pt（≤5 MB，入库）并登记 manifest.json。
  3. 对比：用 LPR_CONFIG 临时指向 v2，跑 eval_accuracy --split val --n 1000 --seed 42 与 --dev-hard --n 300，和 v1 同条件对比整牌、逐位、自动通过率；
     给出 v2 剩余混淆 Top20。目标：val 整牌比 v1 高 ≥5 个百分点。不改 default.toml（切换由架构师做）。
  4. 两种基线（模板、SVM）也在 v2 字符上重训，更新字符级三方法对比表（A5 用）。
只允许新建或修改：algorithm/lpr/recognize/cnn_model.py、cnn.py、template.py、svm.py；algorithm/lpr_tools/train/；algorithm/lpr_tools/eval/ablation.py；
  algorithm/models/char_cnn_v2.pt 与 algorithm/models/manifest.json；algorithm/tests/ 下你名下的测试；docs/design/03-车牌识别.md；D:\keshe2\algorithm\data\chars\。
禁止：改 default.toml 与其他代码；用测试清单或金样例里的图训练或调参；pip install；切换或合并分支；改其他工位。
协作方式：积极派子代理并行（字符重建、训练、基线、分析可并行，训练放后台）；看图先缩小到 ≤400 px，一次不超过 8 张。
汇报方式：完整汇报写到 algorithm/out/receipts/第六批.md；聊天里只回一行：回执 algorithm/out/receipts/第六批.md，commit <最后提交号>
```
