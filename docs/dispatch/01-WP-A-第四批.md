# 派单 01 · 员工 A · 第四批：真实字符样本与 CNN v0（A1、A2、A3）

```text
你是 M2 车牌识别模块执行者【员工 A，工位 01】。工作目录 D:\keshe2\workers\01，分支 feature/m2-wpa-model，用 .venv\Scripts\python.exe。
必读：docs/M2-车牌识别-开发文档.md 7.2—7.4 节（V0.5）、第 8 章、11.2 节、12.5 节、第 15 章
共享数据（不入库，绝对路径）：原始 D:\keshe2\algorithm\data\raw\，索引 D:\keshe2\algorithm\data\index\ccpd_index.csv，
  测试清单副本 D:\keshe2\algorithm\data\index\testset_manifest_v1.csv（03 号员工正在生成，通常 1 小时内出现）。
本批允许写入共享目录：D:\keshe2\algorithm\data\chars\。
任务：
  1. 字符样本（A1）：等测试清单副本出现后（每 5 分钟检查一次；等待期间先做第 4 项），用 build_chars 生成
     D:\keshe2\algorithm\data\chars\v0\，必须加 --exclude-manifest 指向测试清单副本；每类上限 8000；输出每类数量表；随机抽查 50 张标签。
     版式槽用当前 default.toml（02 号员工正在校准，校准后下一批再重建，这次不用等）。
  2. CNN v0（A2）：真实字符 + 合成省份字（synth_chars，补样本少的省份）混合训练，15 轮左右，后台运行并写日志；
     目标验证集字符准确率 ≥95%（省份字单列）。模型存为 algorithm/models/char_cnn_v0.pt（≤5 MB，允许入库）并登记 models/manifest.json，
     把 default.toml 不要改（runtime.model_path 由架构师切换）。
  3. 两种基线（A3）：在同一真实字符集上训练模板与 SVM（--calibrated），模型放 algorithm/out/models/（不入库）；用 ablation 出字符级三方法对比表。
  4. 文档：把 docs/design/03-车牌识别.md 的 3.12 节按开发文档 V0.5 更新（清晰测试图来自 base 验证集留出 120 张、共享数据目录约定）。
只允许新建或修改：algorithm/lpr/recognize/cnn_model.py、cnn.py、template.py、svm.py；algorithm/lpr_tools/train/；algorithm/lpr_tools/eval/ablation.py；
  algorithm/models/char_cnn_v0.pt 与 algorithm/models/manifest.json；algorithm/tests/ 下你名下的测试；docs/design/03-车牌识别.md；D:\keshe2\algorithm\data\chars\。
禁止：改其他代码与 default.toml；用测试清单里的图训练或调参；pip install；切换或合并分支；改其他工位。
协作方式：积极派子代理并行（如训练在后台时并行做基线与文档）；看图先缩小到 ≤400 px，一次不超过 8 张。
汇报方式：完整汇报写到 algorithm/out/receipts/第四批.md；聊天里只回一行：回执 algorithm/out/receipts/第四批.md，commit <最后提交号>
```
