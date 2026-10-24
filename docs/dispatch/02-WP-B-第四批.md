# 派单 02 · 员工 B · 第四批：真实数据版式校准与定位召回（B1、B3）

```text
你是 M2 车牌识别模块执行者【员工 B，工位 02】。工作目录 D:\keshe2\workers\02，分支 feature/m2-wpb-locate，用 .venv\Scripts\python.exe。
必读：docs/M2-车牌识别-开发文档.md 6.1—6.4 节、7.4 节（V0.5）、11.3 节、12.5 节；你的 lpr_tools/calib_layout.py；lpr_tools/eval/eval_accuracy.py 的 --help
共享数据（不入库，绝对路径）：原始 D:\keshe2\algorithm\data\raw\，索引 D:\keshe2\algorithm\data\index\ccpd_index.csv，
  测试清单副本 D:\keshe2\algorithm\data\index\testset_manifest_v1.csv（03 号员工正在生成；生成前不要做任何调参）。
本批允许写入共享目录：D:\keshe2\algorithm\data\calib\。
铁律：调参只用 val 划分和“难例开发集”（ccpd_db/blur/tilt/rotate/fn/challenge/weather 中不在测试清单里的图），绝不用测试清单里的图。
任务：
  1. 版式校准（B3）：用 calib_layout 在 ccpd_base val 2000 张上校准 7 位、在 ccpd_green val 上校准 8 位；人工抽看 10 张校正图核对（缩略图）；
     把校准值写入 default.toml 的 layout7_* / layout8_*（注释写来源与样本数，去掉【临时值】标记）；报告真值定位下分割位数正确率（目标 ≥97%）校准前后对比。
  2. 定位召回（B1）：用 eval_accuracy --locate-recall 在 ccpd_base val 1000 张上测 Top-3 定位召回（目标 ≥90%），ccpd_green val 单独报告；
     未达标就调 preprocess/locate 参数与实现（仿真里绿牌召回只有 31%，重点查绿牌 HSV 范围），每改一次记录前后数字。
  3. 各阶段耗时：在 1280×720 下测预处理、定位、校正、分割的 P95，对照 4.2 预算。
只允许新建或修改：algorithm/lpr/preprocess.py、locate.py、rectify.py、segment.py；algorithm/lpr_tools/synth_scenes.py、synth_ccpd.py、calib_layout.py、viz.py；
  algorithm/config/default.toml 的 [preprocess] [locate] [rectify] [segment] 段的值；algorithm/tests/ 下你名下的测试；D:\keshe2\algorithm\data\calib\。
禁止：改键名和冻结值；改其他代码；pip install；在测试清单的图上调参；切换或合并分支；改其他工位。
协作方式：积极派子代理并行（如校准与召回测量并行）；看图先缩小到 ≤400 px，一次不超过 8 张。
汇报方式：完整汇报写到 algorithm/out/receipts/第四批.md；聊天里只回一行：回执 algorithm/out/receipts/第四批.md，commit <最后提交号>
```
