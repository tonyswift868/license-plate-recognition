# 派单 01 · 员工 A · 第十二批：省份全覆盖——三个读牌器补齐 31 个省份字（真实外省牌均衡采样 + 省份字替换合成）+ 详细设计素材

```text
你是 M2 车牌识别模块执行者【员工 A，工位 01】。工作目录 D:\keshe2\workers\01，分支 feature/m2-wpa-model。
训练用 D:\keshe2\.venv-gpu，推理、校准与测试用你自己的 .venv。
必读：你第十一批回执；03 第十二批回执与 docs/research/03-三路仲裁集成规格.md；02 第十一批回执（plate_det_v1）；algorithm/reports/scoreboard.md 最上面两行；
      docs/M2-车牌识别-开发文档.md 中“数据分布偏斜”风险与“字体合成（已确认 Q7）”一行；docs/research/01-整牌识别调研.md。
背景（架构师 10-28）：
  - 你的 box_reader_v2 已验收合入 develop，架构师在 develop 重跑三路 3100 张与你逐张一致。v2 三路 majority：最暗档 48/51、tilt+rotate 88.0%、绿牌 96.4%。
  - 用户决定：系统要识别本地（陕等）和外地车牌，**训练集必须覆盖全部 31 个省份**。CCPD 全索引省份字分布（含保护名单）：
    皖 337,943（92.9%）；苏 7,881、浙 3,590、沪 1,763、豫 1,426、粤 1,359、鄂 1,164、鲁 1,157、京 1,122、闽 870、赣 809、湘 747、冀 666、陕 606、
    川 441、晋 436、津 419、渝 397、辽 320、云 147、黑 141、甘 98、贵 59、蒙 50、桂 40、新 36、吉 27、青 15、琼 14、宁 9、藏 1。
  - 架构师在 v2 三路 trace 上算了 val1000+devhard 里 261 张非皖牌：省份字读对率 框读 v2 90.8%（皖牌 94.8%）、校正 crnn_v2 75.5%（33 张读成“皖”）、
    分割+CNN 63.6%（14 张读成“皖”）；除省份字外其余位框读 95.8%。即外省牌主要错在省份字，且旧两路一起读成“皖”会被自动通过——三个读牌器都要补。
任务：
  0. 调研（1 小时内，补到调研文档新一节）：同类项目怎样解决省份字不均衡与覆盖——
     公开中文车牌数据集（如 CLPD、CRPD、CBLPRD-330k 等：规模、省份覆盖、许可、下载体积）只列清单、**不下载**；
     合成车牌 / 省份字替换（贴字、渲染、风格匹配）做法；按类别均衡采样。结尾写本批采用什么、外部数据集是否值得申请下载（由架构师问用户）。
  1. 数据（全部在线生成，不落新图片缓存）：
     a. 省份留出：从 CCPD 非皖、且不在五份名单里的图中，按固定种子每省取 min(20, 该省 30%) 张作省份留出，写 algorithm/out/data/batch12/province_holdout.csv
        （含每省张数）；留出不进训练。少于 5 张的省份只用合成评测。
     b. 真实外省牌按省份均衡采样：在你第十一批分组均衡的基础上再按省份字均衡（每省每轮抽样数设上下限，回执写明），其余保持第十一批的组与扰动分布。
     c. 省份字替换合成：在真实训练裁片上按 GT 四角点定位第 1 个字符格，换成渲染的另一个省份字（本机字体如 C:\Windows\Fonts\simhei.ttf，字体不入库），
        匹配底色（蓝/绿/黄）、字色、模糊、噪声与透视，标签同步改；31 个省份均匀覆盖，合成占比自定并写理由。先缩小看 ≤8 张确认像真牌。
     d. 合成评测集：用固定种子在调参集（非五份名单）裁片上生成 31 省 × 30 张，评测时确定性生成，不落盘或只存 ≤50 MB。
  2. 三个读牌器都补省份（都从现有权重继续训练，保住其他位）：
     a. box_reader_v3：从 v2 继续，按第 1 项数据训练至少 20 个完整 epoch，按调参集（含省份留出）选模型，温度校准同前。
     b. crnn_v4（校正读路线）：从 crnn_v2 继续，在 GT 四角点校正裁片上用同样的省份均衡与合成，至少 15 个完整 epoch。
     c. char_cnn_v3（分割+CNN 路线的省份字）：真实外省首字裁片 + 字级渲染合成，至少补齐 31 类；其余类别不退步。
     模型存 algorithm/models/ 并登记 manifest；工厂按 checkpoint 加载新旧版本。
  3. 评测（只报告，不调参）：
     a. 每个读牌器：省份留出（逐省）、合成 31 省集（逐省）、调参集各子集；省份字读对率、整牌读对率、读成“皖”的张数。
     b. 实际候选（与第十一批 2b 同口径，定位用 plate_det_v1——在 out 下复制 default.toml，把 runtime.corner_model_path 改成 models/plate_det_v1.pt）：
        val1000_v1、devhard_v1 各子集 + 非皖 261 张单列。
     c. 三路组合：用 03 的一条命令 three_route_workflow --fresh --mode all，临时配置里换上 crnn_v4、char_cnn_v3、plate_det_v1，--box-model 换 box_reader_v3，
        报 val、绿牌、难例、最暗档、tilt+rotate、非皖、两集合自动通过精度，以及按测试集组成折算。
     目标：省份留出上三路各自省份字 ≥97%，陕 ≥95%（留出 ≥20 张时）；合成 31 省集每省省份字 ≥85%、总体 ≥95%；
           非皖 261 张三路组合整牌不低于皖牌；不退步：val1000、devhard 上各路线整牌与三路组合各项下降不超过 1 个百分点；框读 read 平均 ≤6 ms、P95 ≤8 ms（2 线程）。
  4. 《详细设计》第 3 章素材（10-30 交，架构师整理；可在 GPU 训练期间写）：新建 docs/design/ch3-识别模型.md——框直接读（STN + MobileNetV3 前段 + BiGRU + CTC）、
     校正 CRNN、分割 + CNN 三个读牌器的结构（各一张 mermaid 结构图）、输入输出、训练数据与排除规则、省份均衡与合成、训练设置、开发集结果表、CPU 时延。书面中文，约 3 页，数字只引用已复核的。
  止损：训满第 2 项训练量后，若省份留出上省份字仍低于 90% 且合成集无提升，才停下写原因；不许因前几轮不涨提前停。
资源与磁盘（用户机器，内存紧张）：GPU 与 02 共用，你最多 6 GB 显存；你所有进程合计内存 ≤3 GB；复用第十一批裁片缓存，不建新图片缓存或数据集；
  checkpoint 每个模型只留 best 与 last；本批新增磁盘 ≤2 GB；三个模型训练合计有效墙钟目标 ≤7 小时，按 a→b→c 顺序串行，
  每个模型第 1 个 epoch 后把每 epoch 耗时和推算总时长写进回执草稿。内存不足时先存档再停，在聊天里说明需要多少空闲内存。
只允许新建或修改：algorithm/lpr/recognize/（base.py 除外）；algorithm/lpr_tools/train/；algorithm/lpr_tools/eval/ablation.py；
  algorithm/models/box_reader_v3.pt、crnn_v4.pt、char_cnn_v3.pt 与 algorithm/models/manifest.json（只加你的条目）；
  algorithm/tests/ 下以 test_crnn、test_recognize、test_train、test_box_reader、test_char 开头的文件；docs/research/01-整牌识别调研.md；docs/design/ch3-识别模型.md。
禁止：改 pipeline.py、config.py、boxcrop.py、recognize/base.py、default.toml；车牌专用现成模型/库/权重；下载任何数据集、字体或权重（只列清单）；复制他人代码；安装新包；
  用五份名单（testset、golden、devhard、val1000、undecodable）里的图训练或调参；切换或合并分支；改其他工位。
协作方式：积极派子代理并行；看图先缩小到 ≤400 px，一次不超过 8 张。
汇报方式：完整汇报写到 algorithm/out/receipts/第十二批.md；聊天里只回一行：回执 algorithm/out/receipts/第十二批.md，commit <最后提交号>
```
