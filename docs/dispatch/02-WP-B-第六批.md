# 派单 02 · 员工 B · 第六批：四角点多假设与“分割成功但字框不对”

```text
你是 M2 车牌识别模块执行者【员工 B，工位 02】。工作目录 D:\keshe2\workers\02，分支 feature/m2-wpb-locate，用 .venv\Scripts\python.exe。
必读：algorithm/reports/scoreboard.md 前两行；03 号员工第五批的逐阶段归因（D:\keshe2\workers\03\algorithm\out\receipts\第五批.md 的“同一批 val 1000 正常/Oracle 对照”与混淆 Top20）；
      algorithm/lpr/pipeline.py 里新加的 _rectifications()（架构师已按你的提案改了编排）；开发文档 5.9 节。
现状（develop，cnn v1）：val 1000 整牌 44.6%（base 58.6%，green 30.6%），SEG_FAILED 仍有 481/1000；高频混淆 0→1、D→1、7→1、0→L，像是字框被切窄或切偏。
任务：
  1. 在 rectify.py 实现 rectify_candidates(image, cand, cfg) -> list[RectifiedPlate]：同一候选最多 4 个四角点假设，按像素证据排序，
     第一个必须与 rectify() 的结果一致。流水线已会逐个假设尝试分割，直到出现 OK。只用像素证据，不用标注。
  2. 查“分割成功但字框不对”：在 val 上挑分割成功但识别错的样本，按错误类型（字框切窄、切偏、相邻字粘连、边框/铆钉混入、省份字碎片）分类计数，修 segment 的字框。
  3. 绿牌专项：green val 只有 30.6%，单独列出失败原因与改进前后对比。
  目标（val 1000，eval_accuracy --split val --n 1000 --seed 42，cnn v1）：SEG_FAILED ≤ 250，整牌准确率 ≥ 55%，Top-3 定位召回不下降，G1 的 P95 ≤ 200 ms。
重要约束：本批不要改“字符图怎么归一化”（现在是全高槽位上下文 + normalize_char），只改用哪些框、哪些四角点；01 号员工正按当前归一化重建训练字符，改了会再次错配。
只允许新建或修改：algorithm/lpr/preprocess.py、locate.py、rectify.py、segment.py；algorithm/lpr_tools/synth_scenes.py、synth_ccpd.py、calib_layout.py、viz.py；
  algorithm/config/default.toml 的 [preprocess] [locate] [rectify] [segment] 段的值；algorithm/tests/ 下你名下的测试；D:\keshe2\algorithm\data\calib\。
禁止：改 pipeline.py（编排改动在回执里提）；改键名和冻结值；pip install；在测试清单或金样例上调参；切换或合并分支；改其他工位。
协作方式：积极派子代理并行（如一个做 rectify_candidates，一个做字框错误分类，一个做绿牌）；看图先缩小到 ≤400 px，一次不超过 8 张。
汇报方式：完整汇报写到 algorithm/out/receipts/第六批.md；聊天里只回一行：回执 algorithm/out/receipts/第六批.md，commit <最后提交号>
```
