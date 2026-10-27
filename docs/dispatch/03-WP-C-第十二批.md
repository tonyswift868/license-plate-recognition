# 派单 03 · 员工 C · 第十二批：三路组合的回退规则细化与集成规格

```text
你是 M2 车牌识别模块执行者【员工 C，工位 03】。工作目录 D:\keshe2\workers\03，分支 feature/m2-wpc-post-eval，用 .venv\Scripts\python.exe。
必读：你第十一批回执；架构师用 box_reader_v1 跑的三路组合 D:\keshe2\algorithm\out\tmp_arch\three_v1\report.json 与 live.jsonl.gz；01 第十批回执；
      algorithm/lpr/pipeline.py 当前默认流程（agree + agreement_then_reader）。
背景（架构师 10-27）：
  - 换上 box_reader_v1 后，你的三路规则族在两个开发集自动通过精度 ≥99.5% 下自动通过 1859 张（现默认 1804）。
  - 未自动通过时的回退选法差别很大：majority 让最暗档 58.8%→76.5%、难例总体 68.3%→74.6%，但 val 绿牌 92.6%→87.8%、tilt+rotate 76.3%→75.5%；
    agreement_then_reader 则相反。原因很清楚：框路线读夜间、模糊强，读绿牌（8 位）和倾斜弱。
任务：
  1. 回退规则细化（只改未自动通过时返回哪个读法，不改自动通过集合）：在 three_v1 的 trace 上比较若干预先写定、不读真值的规则，例如
     按候选颜色或版式分流（绿牌、8 位走校正路线）、按框路线与校正路线置信度差分流、按候选几何（四角点推出的倾斜角）分流、按图像亮度分流，以及它们的组合。
     目标：val 绿牌不低于现默认（92.6%），同时最暗档与难例总体尽量接近 majority。为防过拟合：规则参数只许在 val1000 与 devhard 的一半上选（按固定种子切），在另一半上报告；两半都列出。
  2. 自动通过规则复核：在同一 trace 上，给三路规则族加上“同 rect S/R 两路一致”的旧单边门槛是否还能多通过、精度是否守住；仍约束两集合各 ≥99.5%。
  3. 集成规格：把最终推荐的“自动通过规则 + 回退规则”写成架构师可直接照着实现的规格（伪代码、阈值完整浮点值、遍历顺序、平分规则、需要新增的配置键），
     写到 docs/research/03-三路仲裁集成规格.md。配一份逐张期望输出（压缩 JSONL：path、status、plate、confidence、route），架构师实现后用它逐张核对。
  4. 两个工具都要能一条命令换模型重跑：01 的 box_reader_v2、02 的 plate_det_v1 交回后直接复测。
资源与磁盘：不用 GPU；CPU 线程 ≤4；你所有进程合计内存 ≤1 GB，评测一次只跑一个进程；新增磁盘 ≤500 MB，trace 用压缩 JSONL；pytest 临时目录跑完就删。
只允许新建或修改：algorithm/lpr_tools/eval/（ablation.py 除外）；algorithm/tests/ 下以 test_eval、test_three_route、test_fallback 开头的文件；docs/research/03-三路仲裁集成规格.md。
禁止：改运行时代码（algorithm/lpr/）、配置、模型、test_golden.py、scoreboard.md；在测试清单上调参；pip install；切换或合并分支；改其他工位；删除本批自己临时文件以外的任何文件。
协作方式：积极派子代理，但同一时间只跑一个评测进程；看图先缩小到 ≤400 px，一次不超过 8 张。
汇报方式：完整汇报写到 algorithm/out/receipts/第十二批.md；聊天里只回一行：回执 algorithm/out/receipts/第十二批.md，commit <最后提交号>
```
