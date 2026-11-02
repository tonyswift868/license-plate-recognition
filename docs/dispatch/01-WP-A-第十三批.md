# 派单 01 · 员工 A · 第十三批：lpr_tools 与 tests 按团队 ruff 规则清理，测试能在团队仓库布局下跑

```text
你是 M2 车牌识别模块执行者【员工 A，工位 01】。工作目录 D:\keshe2\workers\01，分支 feature/m2-wpa-model（已同步到 develop，含本派单），用 .venv\Scripts\python.exe。

为什么做（架构师 11-03）：M2 要并进团队仓库，组长的后端已上团队 develop，我们是唯一待交付的依赖，11-06 团队集成。
  训练与评测工具 lpr_tools、测试 tests 要一起交（课设要求模型可复现、有三方法对比），所以要满足两条：
  - 过团队的 ruff 检查。规则是 E、F、I、B、UP，行宽 100，py311，忽略 E501。
    配置在 D:\keshe2\workers\parkinglot\pyproject.toml（团队仓库的本地副本，你只读）。
  - 在团队仓库里只带 4 个默认模型、没有 data/ out/ testset/ 的情况下，测试要么通过、要么明确跳过，不能失败。组员拉下来要能直接跑。

本批分工：
  - 你清 algorithm/lpr_tools/ 与 algorithm/tests/ 的已有文件。
  - algorithm/lpr/ 归 02，pipeline.py 归 03。
  - 03 会新建 algorithm/tests/test_backend_contract.py 和 algorithm/lpr_tools/team_export.py，你不碰这两个文件（你的分支里也还没有）。

任务：
  1. 装 ruff：.venv\Scripts\python.exe -m pip install ruff（最新版，只装这一个包，授权联网），回执写版本号。
     在工位根目录先记清理前的统计：
     ruff check --config D:\keshe2\workers\parkinglot\pyproject.toml --statistics algorithm/lpr_tools algorithm/tests
     再在 algorithm/ 下跑一次全量 pytest，作为基线。
  2. 清理：先加 --fix（不加 --unsafe-fixes），再手改剩下的。
     只做不改行为的修改。确实改不了或改了有风险的，在行尾写 # noqa: <规则号>，并注明原因。回执逐条列出，越少越好。
  3. 模拟团队仓库布局：
     a. 在你的 out 目录建临时目录 T。用 git archive 导出你当前提交的这些路径，解到 T 下：
        algorithm/lpr、algorithm/lpr_tools、algorithm/tests、algorithm/config、algorithm/reports/scoreboard.md、algorithm/pyproject.toml
     b. 再把 algorithm/models/ 下的 plate_det_v1.pt、char_cnn_v2.pt、crnn_v4.pt、box_reader_v3.pt、manifest.json 五个文件复制进 T。
     c. 建一个空的 T\algorithm\__init__.py（团队仓库就是这样，algorithm 是个包）。
     d. 在 T\algorithm 下运行 python -m pytest -p no:cacheprovider --basetemp <你的 out 下的目录>。
     失败的测试按原因处理：
     - 缺非默认模型，或缺 data/、out/、testset/ 等本地文件：改成缺文件时 pytest.skip，写明缺什么。沿用你第十二批“缺本地数据就跳过”的做法。
     - 其他原因（例如 algorithm 有 __init__.py 时导入方式出错）：修测试的导入或路径写法，不改 lpr 源码；修不了的列进回执。
     另外搜一遍 tests 和 lpr_tools 里写死的 D:\keshe2 之类绝对路径。在组员机器上这些路径不存在，必须是缺文件就跳过，或改成相对仓库的路径。
  4. 回到我们的完整仓库确认：
     - 全量 pytest 结果与第 1 步基线一致，跳过数不许因为你的改动增加；
     - ruff check --config D:\keshe2\workers\parkinglot\pyproject.toml algorithm/lpr_tools algorithm/tests 输出 0 个问题。

资源与磁盘：不用 GPU；内存 ≤2 GB；测试一次只跑一个进程；新增磁盘 ≤300 MB（T 约 25 MB，测完在回执里列出，由架构师清理）。
  预计 2～3 小时（lpr_tools 99 个文件、tests 134 个文件，量较大）。

只允许修改：algorithm/lpr_tools/ 与 algorithm/tests/ 下的已有文件。

禁止：
  - 改 algorithm/lpr/、config、模型、文档、pyproject.toml。
  - 改 D:\keshe2\workers\parkinglot 下任何文件。
  - 跑冻结测试集；训练；除 ruff 外 pip install。
  - 切换、合并或推送分支；改其他工位。

汇报方式：完整汇报写到 algorithm/out/receipts/第十三批.md，内容包括：
  - 清理前后的规则统计；
  - noqa 清单；
  - T 下测试结果：通过、跳过、失败各多少，跳过按原因分类；
  - 完整仓库 pytest 结果；
  - 临时目录清单。
聊天里只回一行：回执 algorithm/out/receipts/第十三批.md，commit <最后提交号>
```
