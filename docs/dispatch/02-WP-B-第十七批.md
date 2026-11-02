# 派单 02 · 员工 B · 第十七批：algorithm/lpr 按团队 ruff 规则清理（识别结果逐张不变）

```text
你是 M2 车牌识别模块执行者【员工 B，工位 02】。工作目录 D:\keshe2\workers\02，分支 feature/m2-wpb-locate（已同步到 develop，含本派单），用 .venv\Scripts\python.exe。

为什么做（架构师 11-03）：M2 要并进团队仓库，组长的后端已上团队 develop，我们是唯一待交付的依赖，11-06 团队集成。
  团队 PR 硬性要求 ruff check 通过。规则是 E、F、I、B、UP，行宽 100，py311，忽略 E501。
  配置在 D:\keshe2\workers\parkinglot\pyproject.toml（团队仓库的本地副本，你只读）。
  algorithm/lpr 是运行时代码，答辩跑的就是它，所以清理后识别结果必须逐张不变。

本批分工：你清 algorithm/lpr/ 下除 pipeline.py 外的全部已有文件（含 lpr/recognize/）。
  pipeline.py 归 03（它同时在加团队适配函数），lpr_tools/ 与 tests/ 归 01。

任务：
  1. 装 ruff：.venv\Scripts\python.exe -m pip install ruff（最新版，只装这一个包，授权联网），回执写版本号。
  2. 改动前先做两件事：
     a. 在工位根目录运行：
        ruff check --config D:\keshe2\workers\parkinglot\pyproject.toml --statistics --extend-exclude algorithm/lpr/pipeline.py --force-exclude algorithm/lpr
        把清理前各规则的数量记进回执。
     b. 用默认配置在 val1000_v1 前 200 张（按清单顺序）跑一遍，保存逐张输出。用现有 lpr_tools.eval 工具（如 dump、compare），不新写评测框架。
  3. 清理：先加 --fix（不加 --unsafe-fixes），再手改剩下的。
     - 只做不改行为的修改。确实改不了或改了有风险的，在行尾写 # noqa: <规则号>，并注明原因。回执逐条列出，越少越好。
     - 导入顺序重排后可能触发循环导入（尤其 lpr/__init__.py），跑测试确认。
     - 不改 pipeline.py，命令里一直带 --extend-exclude algorithm/lpr/pipeline.py --force-exclude。
  4. 行为零变化验证：
     a. algorithm/ 下全量 pytest 通过，结果与改动前在你工位跑的一致（先跑一次记基线）。
     b. 清理后同样 200 张再跑一遍，与 2b 逐张比较 status、plate_no、confidence（差 ≤1e-6）、candidates，要求 0 差异。
     c. 金样例 20/20。
  5. 结束时第 2a 条的命令（去掉 --statistics）输出 0 个问题。

资源与磁盘：不用 GPU；内存 ≤1.5 GB；评测一次只跑一个进程，torch 2 线程；新增磁盘 ≤200 MB。预计 1～1.5 小时。

只允许修改：algorithm/lpr/ 下除 pipeline.py 以外的已有文件。

禁止：
  - 改 pipeline.py、lpr_tools/、tests/、config、模型、文档。
  - 改 D:\keshe2\workers\parkinglot 下任何文件。
  - 跑冻结测试集；除 ruff 外 pip install。
  - 切换、合并或推送分支；改其他工位。

汇报方式：完整汇报写到 algorithm/out/receipts/第十七批.md，内容包括：
  - 清理前后的规则统计；
  - noqa 清单；
  - 零差异比较的命令与结果；
  - pytest 结果。
聊天里只回一行：回执 algorithm/out/receipts/第十七批.md，commit <最后提交号>
```
