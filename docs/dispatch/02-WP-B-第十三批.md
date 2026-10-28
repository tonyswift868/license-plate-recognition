# 派单 02 · 员工 B · 第十三批：《详细设计》第 3 章按现默认流程更新（10-30 交刘一凡），图 5 源文件

```text
你是 M2 车牌识别模块执行者【员工 B，工位 02】。工作目录 D:\keshe2\workers\02，分支 feature/m2-wpb-locate，用 .venv\Scripts\python.exe。
必读：docs/design/03-车牌识别.md（要改的正文）；docs/design/ch3-定位检测器.md、docs/design/ch3-仲裁与评测.md（develop 上的素材）；
      01 的素材 D:\keshe2\workers\01\docs\design\ch3-识别模型.md（只读，不在你分支上，引用时写明来源）；
      algorithm/reports/scoreboard.md 最上面一行；algorithm/config/default.toml、legacy_agree_v1.toml；docs/M2-接口说明-M3.md。
为什么现在做（架构师 10-28）：第 3 章 10-30 要交刘一凡，正文还停在 10-26（四角点网络 + 分割单路）。现在默认流程已经是
  plate_det_v1 检测 + 三路读牌（分割+CNN、拉正 crnn、框读 box_reader）+ 三路仲裁。01 的省份模型约 11 点交付，数字用占位符。
任务：
  1. 按现默认流程改写 03-车牌识别.md，保留原章节编号习惯和文风，不另起一份文档：
     a. 3.3 总体流程与 3.5—3.9 各阶段：改成现默认流程，旧流程（四角点网络、agree 双路）只在一小节里作为“演进与对照”保留，并写明留档配置 legacy_agree_v1.toml。
     b. 新增或改写神经网络小节：plate_det_v1（结构、输入、输出头、训练数据与排除规则、训练量）、char_cnn、crnn、box_reader（同样四项），
        三路仲裁（011 + T20_A 的判定顺序、阈值来源与 fit/hold 选法、早返回）。
     c. 3.12 数据与 3.13 评测：写清开发集与冻结测试清单的隔离、G1/G2/G3/G7 口径；数字只用开发集和积分板上已验收的值，测试集结果一律写“报告阶段填写”。
     d. 省份：写方案（真实外省均衡 + 省份字替换合成、31 省验证）与 box_reader_v3 已有的数（真实外省留出 97.1%、合成 31 省 88.5%），
        crnn_v4、char_cnn_v3 的数写“【待填：01 第十二批续】”。
     e. 3.14 风险与 3.15 待确认：更新为当前情况（难例自动通过精度差 1 张、摄像头小车牌、G3 视频未到、本地陕牌照片）。
  2. 图 5：按现流程更新 draw.io 源文件（docs/img/src/ 下），节点与 3.3 节文字一致；PNG 导出由用户做，回执写清要导出的文件名。
  3. 每个数字后注明出处（积分板行、回执文件名或 commit），没有出处的数字不写。
资源与磁盘：不用 GPU，不跑模型、不跑评测；内存 ≤0.5 GB；新增磁盘 ≤20 MB。预计 2—2.5 小时。
只允许新建或修改：docs/design/03-车牌识别.md、docs/img/src/ 下图 5 源文件、docs/design/ch3-定位检测器.md。
禁止：改代码、配置、模型、测试、scoreboard.md；编造或估算未验收的数字；切换或合并分支；改其他工位。
协作方式：可以派子代理分节起草，最后由你统一术语和编号。
汇报方式：完整汇报写到 algorithm/out/receipts/第十三批.md（列出改了哪些节、待填占位符清单、图 5 导出说明）；聊天里只回一行：回执 algorithm/out/receipts/第十三批.md，commit <最后提交号>
```
