# 派单 03 · 员工 C（WP-C 后处理·多帧·评测）· 首批

```text
你是“停车场收费系统”M2 车牌识别模块的执行者【员工 C / WP-C，工位 03】。
工作目录：D:\keshe2\workers\03    分支：feature/m2-wpc-post-eval（已检出，不要切换分支）
环境：工作目录自带 .venv（Python 3.12，依赖已装好，lpr 已可编辑安装）。所有命令在工作目录根下运行，
      用 .venv\Scripts\python.exe，例如 .venv\Scripts\python.exe -m pytest algorithm/tests -q
必读：docs/M2-车牌识别-开发文档.md 第 2、5 章（含 5.9），6.6—6.8 节，7.4 节，11.4 节，第 13 章，附录 A、B；
      algorithm/lpr/types.py、charset.py、errors.py、config.py、pipeline.py、testing.py；algorithm/config/default.toml
背景：CCPD 数据集还没到位。本批任务全部用构造数据完成；不要等数据，也不要自己下载任何数据集。

本次任务（按顺序做，每做完一项提交一次）：
  C-1 后处理（任务 C3，可按文档标准完整验收）
      - plate_rules.py：位置约束集合 PROV/LETTER/ALNUM/DIGIT/DF（基于 charset 的分组下标），
        版式 L7、L8S、L8L 的逐位约束与正则（附录 A），match_layout(plate)，vehicle_type_hint(color, layout_name)（5.8 表）。
      - postprocess.py：decode(probs, layout, color, cfg) 严格按 6.6 与 5.4：逐位在约束集合内取 argmax，
        逐位置信度取原始 softmax 概率（不重新归一化）；8 位在 L8S、L8L 两个假设里按 Σlog p 选优；
        confidence = min(逐位置信度)；Top-k 备选（k = cfg.topk）用 beam search；正则不符 → INVALID_FORMAT，
        达到 cfg.min_confidence → OK，否则 LOW_CONFIDENCE；province_prior 非空时对首位省份概率乘系数（仅用于选字，
        置信度仍取原始概率）；返回 types.DecodeResult（含 layout_name 与 vehicle_type_hint）。
      - tests/test_plate_rules.py、tests/test_postprocess.py：覆盖三种版式、I/O 字母不出现、8 位两种假设都能选中、
        三种状态、Top-k 排序、车型映射全部 5 行。
  C-2 多帧投票与去重（任务 C5，可按文档标准完整验收）
      - voting.py：vote() 严格按 6.7；PlateVoteSession 按 6.7 流式规则（agree_frames 帧一致或超过 window_s 输出一次并重置），
        内部加锁。
      - dedup.py：DuplicateFilter 严格按 6.8（same_plate；编辑距离 ≤1 且校正图 dHash 汉明距离 ≤ dhash_max_hamming
        → similar_plate），内部加锁；dHash 用 OpenCV/NumPy 自己实现。
      - tests/test_voting.py、tests/test_dedup.py：3 帧里 1 帧错能被纠正；有效帧不足 min_frames 时返回最佳单帧；
        layout 众数过滤；流式会话一致输出与超时输出；30 s 内同牌判重、超过窗口不判重、不同车道不判重、similar_plate。
  C-3 CCPD 解析（任务 C0，原架构师任务改派）
      - lpr_tools/data/ccpd.py，按附录 B 与第 15 章：
        · 解码表（与 CCPD 官方一致）：
          provinces = 本项目 charset.PROVINCES 的 31 个 + ["警", "学", "O"]（共 34）
          alphabets = list("ABCDEFGHJKLMNPQRSTUVWXYZ") + ["O"]（共 25）
          ads = list("ABCDEFGHJKLMNPQRSTUVWXYZ0123456789") + ["O"]（共 35）
          第 1 位查 provinces，第 2 位查 alphabets，其余位查 ads；"O" 是占位符，含占位符的样本跳过；
          首位为“警”“学”的样本标记为不支持并跳过。
        · parse_ccpd_name(name) → 记录：plate_no、layout（7/8）、quad（把文件名里的 RB,LB,LT,RT 转成 LT,RT,RB,LB）、
          bbox（x,y,w,h）、tilt_h、tilt_v、brightness、blurriness、area（原值）、skip_reason；格式不对抛 ValueError。
        · build_index(raw_root, out_csv)：遍历 raw/ccpd2019（子目录 ccpd_base、ccpd_db… 与 splits/*.txt：
          train.txt→train、val.txt→val、test.txt 与其他 ccpd_*.txt→test）和 raw/ccpd2020/ccpd_green/{train,val,test}；
          ccpd_np 等无法解析文件名的图记为 has_plate=0；CCPD2020 的 plate_color 为 green，其余为 blue；
          CSV 用 UTF-8（无 BOM），列：path,dataset,subset,split,has_plate,plate_no,layout,plate_color,quad,bbox,
          tilt_h,tilt_v,brightness,blurriness,area,skip_reason；quad 写成 "x1 y1 x2 y2 x3 y3 x4 y4"，bbox 写成 "x y w h"；
          结束时打印各子集、各划分的数量与跳过原因统计。另提供 read_index(path)。
        · 命令：python -m lpr_tools.data.ccpd --raw algorithm/data/raw --out algorithm/data/index/ccpd_index.csv
        · 目录结构与 area 字段含义【待核实】：数据到位后由你复核，代码里集中写成常量便于改。
      - tests/test_ccpd.py：附录 B 示例
        025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15.jpg
        必须解析为 皖AY339S，quad = [(154,383),(363,402),(386,473),(177,454)]，bbox = (154,383,232,90)；
        再构造 8 位绿牌、含占位符、“警”字首位三种文件名；在 tmp_path 里放空文件模拟两套目录结构测 build_index。
  C-4 测试集清单与评测脚本 v0（任务 C1 的代码部分 + 任务 C2 v0）
      - lpr_tools/data/manifest.py：清单 algorithm/testset/manifest.csv 的读写与校验
        （列 path,plate_no,plate_color,category,source,quad,note；category ∈ clear/night/tilt/dirty/local/np；
        np 行 plate_no 为空，其余行车牌须通过 plate_rules 校验；路径不重复；可选检查文件存在）。
        本批不生成正式清单（抽样要等 CCPD），只写代码和单测。
      - lpr_tools/eval/eval_accuracy.py v0：参数 --manifest --split --fake --uniform --classifier --out --scoreboard；
        按 7.4 口径输出分子类整牌准确率、字符准确率、自动通过率与自动通过精度、失败归因（定位/分割/识别/拒识）、
        导出失败样例图到 --out 目录；--oracle 模式用 quad 真值跳过阶段二、三；--scoreboard 时往
        algorithm/reports/scoreboard.md 追加一行（本批只在临时副本上演示，不改真的积分板）。
        --split val 的抽样依赖 CCPD 索引，本批先保留参数并给出“需要 ccpd_index.csv”的提示。
      - tests/test_manifest.py、tests/test_eval_accuracy.py：用 tmp_path 里构造的小清单 + FakeRecognizer 跑通。
  C-5 基准脚本（任务 C4）
      - lpr_tools/eval/bench_latency.py：按 13.2（补边缩放到 --size、预热 5 次、计时 --n 次、P50/P95/最大值、
        各阶段耗时、CPU 型号与 torch 线程数）；支持 --fake / --uniform / --manifest / --images。
      - lpr_tools/eval/bench_video.py：按 13.2 与 Q5 口径（解码全部帧，每秒抽 --sample-fps 帧识别并用 PlateVoteSession
        投票；端到端吞吐 = 解码帧数 ÷ 墙钟时间）；支持 --fake / --uniform。
      - 冒烟：用 cv2.VideoWriter 在 algorithm/out/ 下生成一段 10 秒合成视频来跑。

只允许新建或修改：
  algorithm/lpr/plate_rules.py、postprocess.py、voting.py、dedup.py
  algorithm/lpr_tools/data/ccpd.py、algorithm/lpr_tools/data/manifest.py
  algorithm/lpr_tools/eval/ 下的文件（ablation.py 除外）
  algorithm/testset/ 下的文件
  algorithm/tests/test_plate_rules.py、test_postprocess.py、test_voting.py、test_dedup.py、test_ccpd.py、
    test_manifest.py、test_eval_accuracy.py、test_bench.py
  algorithm/notebooks/ 下的文件
禁止：
  - 修改 types.py、charset.py、charnorm.py、errors.py、config.py、pipeline.py、testing.py、cli.py、
    recognize/ 下任何文件、tests/conftest.py、tests/test_contracts.py、tests/test_pipeline.py
  - 修改 default.toml（需要新键或改值时在汇报里申请）
  - 引入白名单外的依赖（见 algorithm/requirements.txt，不允许 pandas），不得 pip install 任何东西；
    调用任何车牌识别 API 或使用预训练车牌模型
  - 提交 algorithm/data/、algorithm/out/ 下的任何文件；修改 algorithm/reports/scoreboard.md
  - 切换或合并分支，改动 D:\keshe2 主检出或其他 workers 目录
提交：只在本分支提交，提交信息用 feat(m2): / test(m2): / fix(m2): 前缀。

验收命令（必须实际运行，并贴出完整原始输出）：
  .venv\Scripts\python.exe -m pytest algorithm/tests -q
  .venv\Scripts\python.exe -m pytest algorithm/tests --cov=lpr.postprocess --cov=lpr.plate_rules --cov=lpr.voting --cov=lpr.dedup --cov-report=term-missing
  .venv\Scripts\python.exe -m lpr_tools.data.ccpd --help
  .venv\Scripts\python.exe -m lpr_tools.eval.eval_accuracy --help
  .venv\Scripts\python.exe -m lpr_tools.eval.bench_latency --fake --n 20 --size 1280x720
  .venv\Scripts\python.exe -m lpr_tools.eval.bench_video --fake --video algorithm/out/synth.mp4 --sample-fps 5
  另把 eval_accuracy 在构造小清单 + --fake 上的完整输出贴出。

汇报格式：
  1. 变更文件列表（以及最后一个 commit 号）
  2. 实现要点（不超过 10 行）
  3. 参数初值及来源
  4. 验收命令原始输出
  5. 已知问题与未完成项
  6. 需要架构师决策的事项
```
