# 图 5：车牌识别六阶段流水线

图源：[src/pipeline.drawio](src/pipeline.drawio)。成图：[pipeline.png](pipeline.png)。离线渲染脚本：[src/render_drawio.py](src/render_drawio.py)。PNG由脚本读取图源的mxGraph XML生成，不需要draw.io软件、浏览器或网络服务。2026-10-28第十五批已生成；正文3.15的完成记录是当前图交付状态。

## 当前默认与流程

当前默认为 `plate_det_v1`（检测输入384×384）、S路 `char_cnn_v2`、R路 `crnn_v4`、B路 `box_reader_v3`，采用011 + T20_A。模型与参数以 `algorithm/config/default.toml`、正文3.9.1及03第十七批回执为准。图源只更新标签，节点、连线、坐标和折点布局保持不变。

下表与《详细设计》第3章3.3节的输入、输出及衔接一致；引用标记的完整出处见正文3.15.2。

| 阶段 | 输入 | 当前默认输出／衔接 |
| --- | --- | --- |
| 1 预处理 | 原图、车道 ROI | 工作 BGR、增强灰度、HSV、scale 和 offset；原图仍供网络检测和裁片 [配][流] |
| 2 定位 | 原图供 plate_det_v1，工作图供传统定位 | 网络优先、传统补位的融合候选，不是纯网络候选 [流] |
| 3 校正与框裁片 | 当前候选及原图 | 有序透视视图供 S/R；原 bbox 外扩 crop 供 B、每候选读取一次并缓存 [流][裁] |
| 4 分割 | 当前校正视图及长度假设 | 仅 S 路产生字符格；失败不阻断 R/B [流] |
| 5 识别 | 字符格／校正整牌／原框 crop | S/R/B 各自文本、逐字概率与置信度，保留候选／视图身份 [识][流] |
| 6 后处理 | 同候选的路线读法与预测角点 | 011 自动早返回；耗尽后业务门禁和 T20_A 人工回退 [配][仲] |

融合候选保持网络在前、传统补位，重叠去重；按该顺序逐候选、逐校正视图读取。B从当前候选的原框读取一次，在该候选的视图内复用。每个视图内先比较合法同串S/R、S/B、R/B，再检查S-solo、R-solo；首个自动事件返回OK。该候选全部视图结束后才检查B-solo。S自身OK只结束当前视图的版式循环，仍须读取R并执行三路仲裁。

自动搜索耗尽后，首候选首视图的白／黑牌走UNSUPPORTED；其余按候选0预测角点选择T20_A人工回退。有合法读法保留LOW_CONFIDENCE，不把人工回退升级为自动OK。无候选为NO_PLATE，输入无效为BAD_INPUT，非预期阶段异常为INTERNAL_ERROR。不同候选不互投票；同车多帧投票和DuplicateFilter由调用方选择，OK不替代业务放行校验。

图中“四阈值”说明框引用正文3.9.1及03第十七批，精确值如下（来源：当前default.toml、第十七批“全集收紧后默认”表）：

| 门槛 | 精确值 |
| --- | --- |
| agree：three_route_agree_min_confidence | 0.7233636975288391 |
| S：solo_min_confidence_segment | 0.9719608426094055 |
| R：solo_min_confidence_reader | 0.991486668586731 |
| B：three_route_box_solo_min_confidence | 0.9802792072296143 |

各门槛比较含等号，不截断浮点。图示沿用实际早返回逻辑，不把最高分全图选择、去掉传统融合等未实施建议写成现有能力。蓝色表示主要处理，黄色表示判断／回退，绿色表示接受／结果，红色表示失败／人工，紫色表示调用方可选能力，灰色表示身份约束和出处。

## 离线重新生成

从本工位根目录 `D:\keshe2\workers\02` 执行：

```powershell
# 小型几何检查：只在内存中运行，不读取模型或生成文件。
.\.venv\Scripts\python.exe -B docs/img/src/render_drawio.py --self-check

# 默认读src/pipeline.drawio，生成docs/img/pipeline.png。
.\.venv\Scripts\python.exe -B docs/img/src/render_drawio.py
```

依赖仅Python标准库及已装的Pillow 13.0；字体固定默认为 `C:\Windows\Fonts\simhei.ttf`（黑体）。默认宽2400像素、高度按图源画布比例计算为3889像素，白底RGB，PNG写入150dpi元数据。`--width`仅允许2000—2600，避免超出轻量内存预算；字体缺失会报错，不悄悄替换成缺中文的字体。无需安装新包、联网或下载draw.io。

脚本按mxGeometry绘制矩形、圆角矩形、菱形，保留填充色和描边色；按端口与waypoints补齐正交折线，绘制箭头、虚线和连线标签。标题和图注也是图源节点，一并绘制。XML中的 `&#10;` 自动解析为换行；标签过长时自动换行并逐级缩字号，所有文字墨迹框必须落在对应矩形／菱形内。连线标签优先放在不碰节点、不碰任何连线的位置，不用遮住文字来掩盖冲突。

每次生成前检查文字边界、连线穿字与连线标签重叠；失败则报错，不覆盖成图。标准输出JSON包含节点数、连线数、文字字号、PNG尺寸／大小、图源及成图SHA256。当前图源有29个节点（含标题、图注、说明框）、33条连线，其中29条有文字；实际PNG大小和SHA见本批回执。几何自检覆盖矩形／菱形边界、线段相交、折点保持及中文换行／缩字号。脚本仅支持本图使用的平面、未压缩mxGraph子集，不是通用draw.io导出器。

固定章节引用保持：

```markdown
![图 5 车牌识别六阶段流水线](docs/img/pipeline.png)
（draw.io 源文件：docs/img/src/pipeline.drawio）
```

## 旧流程素材

`stages/`保留为**旧流程素材**，当前第3章3.3及pipeline.drawio均未嵌入这些位图；它们不参与此次PNG渲染，也不代表最终默认模型的中间结果。保留原文件，不重新运行模型或生成器。

| 旧素材 | 历史用途 |
| --- | --- |
| [01-original.png](stages/01-original.png) | 仿CCPD原图 |
| [02-color-mask.png](stages/02-color-mask.png) | 传统蓝色掩膜 |
| [03-edge-mask.png](stages/03-edge-mask.png) | 传统Sobel边缘掩膜 |
| [04-candidates.png](stages/04-candidates.png) | 传统候选轮廓与分数 |
| [05-rectified.png](stages/05-rectified.png) | 候选透视校正图 |
| [06-binary.png](stages/06-binary.png) | 二值字符图 |
| [07-segments.png](stages/07-segments.png) | 七位分割框 |
| [08-characters.png](stages/08-characters.png) | 字符格拼图 |

历史素材来自仿真蓝牌 `新ESHGKT`，来源记录沿用原README：

```text
algorithm/data/raw_mock/ccpd2019/ccpd_base/071-92_98-139&572_579&731-565&731_139&708_157&572_579&590-30_4_16_7_6_9_17-93-18.jpg
```

这些素材只说明当时的处理阶段，不能证明正式识别成绩；当前结果及多车牌目标风险以正文3.13／3.14为准。
