# 派单 02 · 员工 B · 第五批：端到端分割（真实流水线里的校正 + 分割）

```text
你是 M2 车牌识别模块执行者【员工 B，工位 02】。工作目录 D:\keshe2\workers\02，分支 feature/m2-wpb-locate，用 .venv\Scripts\python.exe。
必读：algorithm/reports/scoreboard.md 第一行；架构师首次端到端评测输出 D:\keshe2\algorithm\out\day1\val\report.json（逐图 failures）
      与 D:\keshe2\algorithm\out\day1\hard\report.json；docs/M2-车牌识别-开发文档.md 4.4、6.2—6.4 节、12.5 节。
现状：你上批用“真值四角点”测的分割正确率是 95%，但真实流水线（定位框 → 校正 → 分割）里，val 1000 张有 654 张 SEG_FAILED，
      整牌准确率只有 2.4%。金样例 20 张全错，其中 1 张夜间图被误判成白/黑牌而返回 UNSUPPORTED。
铁律：只在 val 和难例开发集上调参（eval_accuracy 默认已剔除测试清单与金样例）；绝不看测试清单的图。
任务：
  1. 量化差距：在同一批 val 图上分别统计“真值四角点 → 校正 → 分割成功率”和“定位候选 → 校正 → 分割成功率”，
     按失败原因分类（四角点精修偏、颜色误判、版式 7/8 判错、分割质量阈值拒绝等），每类给 2—3 个例子（缩略图 ≤400 px）。
  2. 修复 rectify / segment（必要时 locate 的候选几何），目标：val（base、green 各 500，eval_accuracy --split val --n 1000）上
     SEG_FAILED 比例 < 10%，UNSUPPORTED 只在确有白/黑底时出现；同时 Top-3 定位召回不下降（≥89%）。
     蓝牌被判成 8 位、或 7 位牌输出 8 个字符的情况要专门查（流水线对 yellow/unknown 会追加另一种版式）。
  3. 每改一轮都用同一批 val 记录：SEG_FAILED 数、整牌准确率（用 cnn v0）、定位召回；最终给前后对比表。
只允许新建或修改：algorithm/lpr/preprocess.py、locate.py、rectify.py、segment.py；algorithm/lpr_tools/synth_scenes.py、synth_ccpd.py、calib_layout.py、viz.py；
  algorithm/config/default.toml 的 [preprocess] [locate] [rectify] [segment] 段的值；algorithm/tests/ 下你名下的测试；D:\keshe2\algorithm\data\calib\。
禁止：改键名和冻结值；改流水线编排 pipeline.py（需要改编排逻辑就在回执里提出，由架构师改）；pip install；在测试清单或金样例上调参；切换或合并分支；改其他工位。
协作方式：积极派子代理并行（如一个查定位候选几何、一个查分割拒绝原因）；看图先缩小到 ≤400 px，一次不超过 8 张。
汇报方式：完整汇报写到 algorithm/out/receipts/第五批.md；聊天里只回一行：回执 algorithm/out/receipts/第五批.md，commit <最后提交号>
```
