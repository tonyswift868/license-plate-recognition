# 派单 02 · 员工 B · 第十五批：把图 5 渲染成 PNG（不依赖 draw.io 软件），更新图说明

```text
你是 M2 车牌识别模块执行者【员工 B，工位 02】。工作目录 D:\keshe2\workers\02，分支 feature/m2-wpb-locate（已同步到 develop，含本派单），用 .venv\Scripts\python.exe。
必读：docs/img/src/pipeline.drawio（你第十三批改的图源）；docs/img/README-图5.md（还是 10-25 旧流程的说明）；docs/design/03-车牌识别.md 第 83 行图 5 引用与 3.15 “图5版本标签与PNG导出”一行。
为什么做（架构师 10-28）：第 3 章第 83 行引用 docs/img/pipeline.png，但这个文件从来没生成过，文档里的图 5 现在是空的。
  用户机器上没有 draw.io 软件，也不会用它导出，所以改由你用脚本渲染。第 3 章 10-30 交刘一凡，要带上这张流程图。
任务：
  1. 先核对图源节点与最终默认一致：R 路 crnn_v4、B 路 box_reader_v3、检测 plate_det_v1 输入 384、四阈值数值或引用。不一致就改 drawio 里的标签文字，布局不变。
  2. 写 docs/img/src/render_drawio.py：只用标准库与已装的 Pillow（13.0），读 pipeline.drawio 的 mxGraph XML，按 mxGeometry 画出：
     矩形、圆角矩形、菱形节点及其填充色、描边色，多行中文标签（&#10; 换行，字体 C:\Windows\Fonts\simhei.ttf，自动缩字号以放进框内），
     连线（含折点 waypoints、箭头、连线上的文字），以及标题、图注。输出 docs/img/pipeline.png，宽约 2000—2600 像素，白底，文字清晰不重叠。
     不装新包，不联网，不下载 draw.io。
  3. 自查：渲染后检查每个节点的文字都画在框内、没有文字被截断、没有连线穿过文字。在回执里列出节点数、连线数、PNG 尺寸与大小。
  4. 更新 docs/img/README-图5.md：改成现默认三路流程的说明，表格与第 3 章 3.3 节一致；写明 PNG 由 render_drawio.py 从图源生成，以及重新生成的命令。
     旧的六阶段素材（stages/ 目录）若仍用于第 3 章就保留并注明，否则在 README 里标为“旧流程素材”。
  5. 第 3 章 3.15 “图5版本标签与PNG导出”一行改为已完成，并写明出处。
资源与磁盘：不用 GPU，不跑模型；内存 ≤0.5 GB；新增磁盘 ≤10 MB。预计 1—1.5 小时。
只允许新建或修改：docs/img/src/pipeline.drawio（只改标签文字）、docs/img/src/render_drawio.py（新建）、docs/img/pipeline.png（新建）、
  docs/img/README-图5.md、docs/design/03-车牌识别.md（只改 3.15 那一行）。
禁止：改代码、配置、模型、测试、scoreboard.md；pip install；联网；切换或合并分支；改其他工位。
汇报方式：完整汇报写到 algorithm/out/receipts/第十五批.md；聊天里只回一行：回执 algorithm/out/receipts/第十五批.md，commit <最后提交号>
```
