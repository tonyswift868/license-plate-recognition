# 派单 03 · 员工 C · 第八批：难例整牌数据、E1 阈值重算、每日评测

```text
你是 M2 车牌识别模块执行者【员工 C，工位 03】。工作目录 D:\keshe2\workers\03，分支 feature/m2-wpc-post-eval，用 .venv\Scripts\python.exe。
必读：algorithm/reports/scoreboard.md 最新一行；docs/M2-车牌识别-开发文档.md 5.4 节、7.4 节、12.5 节、14 章 K1。
架构师决定（新）：难例子集可以用于训练，但必须排除以下四份名单中的图（路径与同字节副本都要排除）：
  D:\keshe2\algorithm\data\index\testset_manifest_v1.csv、algorithm/tests/data/golden/golden_manifest.csv、
  D:\keshe2\algorithm\data\index\devhard_v1.csv（每日难例评测固定 2100 张）、D:\keshe2\algorithm\data\index\undecodable.csv。
  另：每日验证固定 1000 张已存为 D:\keshe2\algorithm\data\index\val1000_v1.csv，daily.py 与 eval_accuracy 改为直接读这两份固定名单（不再每次重抽），结果可逐日对比。
任务：
  1. 难例整牌数据 plates/v2：在 plates/v1 基础上加入“难例训练池”（db/blur/tilt/rotate/fn/challenge/weather 排除上面四份名单）的整牌图，
     真值四角点校正版与真实流水线候选校正版各一份（流水线要开四角点网络：LPR_CONFIG 设 corner_model_path = "models/corner_net_v1.pt"）。
     输出 D:\keshe2\algorithm\data\plates\v2\，labels.csv 同 v1 列格式，source 标明子集；验证集只用 val 划分，不含难例训练池。CPU 进程 ≤6。
  2. 每日评测改为固定名单（见上），默认配置加上 --config 运行一次 daily.py，产出积分板行（不改 scoreboard.md）。
  3. E1 阈值重算：用当前默认流水线（以 default.toml 为准）在 val1000_v1 上重新扫描 min_confidence，给出自动通过精度 ≥99% 下自动通过率最高的阈值；
     另给一个精度 ≥99.5% 的保守值。只用验证集，不用测试清单。
只允许新建或修改：algorithm/lpr_tools/data/、algorithm/lpr_tools/eval/（ablation.py 除外）、algorithm/testset/、
  algorithm/tests/ 下以 test_ccpd、test_manifest、test_eval、test_bench、test_sample、test_golden_pick、test_daily、test_build_plates 开头的文件；D:\keshe2\algorithm\data\plates\。
禁止：改运行时代码、配置、test_golden.py、scoreboard.md；在测试清单上调参；pip install；切换或合并分支；改其他工位。
协作方式：积极派子代理并行（数据生成在后台跑时并行做第 2、3 项）；看图先缩小到 ≤400 px，一次不超过 8 张。
汇报方式：完整汇报写到 algorithm/out/receipts/第八批.md；聊天里只回一行：回执 algorithm/out/receipts/第八批.md，commit <最后提交号>
```
