# 图 5：车牌识别六阶段流水线

可编辑源文件：[src/pipeline.drawio](src/pipeline.drawio)。图中所有节点、连线和中文标签均为 draw.io 原生对象，没有远端图片依赖。配色区分六阶段主链、判断/回退、可选旁路及输出。

## 图的构成

| 阶段 | 输入 | 处理及输出 |
|---|---|---|
| ① 预处理 | BGR uint8 原图、可选车道 ROI | 输入校验、ROI 裁剪、缩放、CLAHE 与 HSV；输出 PreprocessOut，保留原图坐标映射 |
| ② 定位 | PreprocessOut | HSV 颜色和 Sobel-x 边缘双分支、形态学、轮廓/几何过滤、打分、NMS；输出按分数排序的至多 3 个 PlateCandidate |
| ③ 校正 | 原图、当前候选 | 精修四角点、透视变换、颜色和版式判断；输出 RectifiedPlate，7 位 252×80、8 位 276×80 |
| ④ 分割 | RectifiedPlate、当前 7/8 位版式 | 极性统一、二值化、边框/铆钉清理、槽位对齐与投影细化；输出 SegmentResult，含字符框及 N×32×32 字符图 |
| ⑤ 识别 | N×32×32 字符图 | CharCNN，可切换模板匹配或 HOG+SVM；输出 N×65 概率矩阵 |
| ⑥ 后处理 | 字符概率、版式、颜色 | 位置约束解码、正则检查、置信度与备选排序；输出 DecodeResult，装配为 RecognitionResult |

候选按定位分数依次尝试。每个候选先校正，再构建版式队列：已知版式先试对应位数，未知版式试 7/8 位，黄色或未知颜色补试另一位数。分割失败或解码非 OK 时继续剩余版式，版式耗尽后回退至下一候选；第一个 OK 提前返回。全部耗尽时，先检查首候选校正后的白/黑色特种车牌，再返回最佳非 OK 结果或 SEG_FAILED。没有候选返回 NO_PLATE，输入无效返回 BAD_INPUT，阶段异常转成 INTERNAL_ERROR。

RecognitionResult 汇总状态/错误码、车牌和颜色、位数、置信度及逐位置信度、备选、原图 bbox/corners、校正图和耗时等。候选坐标是原图像素坐标；分割框坐标属于校正图。整牌置信度为选中字符的逐位原始 softmax 概率最小值，不按位置约束集合重新归一化。

多帧处理是可选旁路：recognize_many 对同一车辆的每帧执行主链后逐位投票；流式调用方也可使用 PlateVoteSession。DuplicateFilter 由调用方按车道和时间选择调用，返回重复标记及原因，保留识别结果，业务放行仍由上层决定。

## 素材及复现

这套素材取自**同一张仿 CCPD 蓝牌图**，真值车牌为 `新ESHGKT`。当前定位输出 1 个候选，候选 0 的 L7 分割成功，输出 7 个 32×32 字符。素材到分割为止，不表示分类器已经正确识别出车牌。

来源文件（相对项目根目录）：

```text
algorithm/data/raw_mock/ccpd2019/ccpd_base/071-92_98-139&572_579&731-565&731_139&708_157&572_579&590-30_4_16_7_6_9_17-93-18.jpg
```

| 素材 | 来源 / 所示内容 | 尺寸（宽×高） | 大小（字节） |
|---|---|---:|---:|
| [01-original.png](stages/01-original.png) | 仿真原图；720×1160 缩小 | 450×725 | 344,798 |
| [02-color-mask.png](stages/02-color-mask.png) | viz 的 03_mask_blue；蓝色掩膜经形态学处理 | 450×725 | 2,185 |
| [03-edge-mask.png](stages/03-edge-mask.png) | viz 的 03_mask_edge；Sobel-x、Otsu 与形态学掩膜 | 450×725 | 3,555 |
| [04-candidates.png](stages/04-candidates.png) | viz 的 04_candidates；候选轮廓、颜色和分数 | 450×725 | 347,858 |
| [05-rectified.png](stages/05-rectified.png) | viz 的 05_plate_0；候选 0 的透视校正图 | 252×80 | 37,421 |
| [06-binary.png](stages/06-binary.png) | viz 的 06_binary_0；统一极性并清理后的二值图 | 252×80 | 1,464 |
| [07-segments.png](stages/07-segments.png) | viz 的 07_boxes_0_L7；7 位分割框 | 252×80 | 38,466 |
| [08-characters.png](stages/08-characters.png) | viz 的 08_chars_0_L7；7 张 32×32 字符横向排列 | 224×32 | 2,676 |

8 张 PNG 总计 **778,423 字节（约 0.74 MiB）**，每张宽度 ≤450 px，满足宽 ≤800 px、总大小 ≤2 MB。原图与候选图使用 INTER_AREA 缩小；两张掩膜使用 INTER_NEAREST 保留 0/255 二值；其他小图保持原尺寸。这些素材单独用于说明各阶段，可在 draw.io 编辑时按需插入，当前源图不绑定外部位图。

以下命令在 `D:\keshe2\workers\02` 根目录使用本机 `.venv`，不需要训练模型。

```powershell
# 精确复现本次素材时使用原来的规模；已有 raw_mock 则跳过生成。
if (-not (Test-Path 'algorithm/data/raw_mock')) {
    .venv\Scripts\python.exe -m lpr_tools.synth_ccpd --out algorithm/data/raw_mock --per-subset 50 --seed 42
}

# 保留单引号，避免 PowerShell 将文件名里的 & 解释成操作符。
$sample = 'algorithm/data/raw_mock/ccpd2019/ccpd_base/071-92_98-139&572_579&731-565&731_139&708_157&572_579&590-30_4_16_7_6_9_17-93-18.jpg'
.venv\Scripts\python.exe -m lpr_tools.viz $sample --out algorithm/out/viz-third
```

本次使用第二批保留的 `--per-subset 50 --seed 42` 数据；生成器按规模消耗随机数，精确复现上述文件时应使用该参数。派单指定的备用规模 20 可生成另一套可用素材，需重新选择分割成功的图并更新来源。生成前确认输出根没有要保留的非托管文件，生成器会拒绝覆盖非托管文件。

viz 为该图输出 11 张 PNG（其中包含工作图、灰度图和其他颜色掩膜）。上表选取其中 7 张并另补源图；原始输出位于 `algorithm/out/viz-third/<编号_图像名>/`，该目录不入库。缩小后的可提交素材仅保存在 stages/。


## 导出 150 dpi PNG

本批交付可编辑源文件和阶段素材；pipeline.png 由下面步骤导出。

1. 打开 [app.diagrams.net](https://app.diagrams.net/)，选择“设备”或“以后再说”，通过“文件 → 从…打开 → 设备”选择本目录的 src/pipeline.drawio。
2. 检查中文字体和完整页面，建议保留 Microsoft YaHei（微软雅黑）；需要调整时编辑原生节点和连线。源图可独立打开，阶段素材通过本文单独引用。
3. 选择“文件 → 导出为 → PNG”，展开底部的“高级（Advanced）”，将 **DPI 设为 150**。导出整个图、保留白色背景，并设置少量边距；可勾选“包含图表副本”保留嵌入的可编辑数据。该 DPI 选项位置见 [draw.io 官方 PNG 导出说明](https://www.drawio.com/docs/manual/export/export-to-png/)。
4. 将结果保存为 **docs/img/pipeline.png**。检查边缘没有裁切、中文清楚、回退与旁路连线完整；另存并保留 src/pipeline.drawio 以便后续修改。

在项目文档中遵循《AI 开发说明》11.3 的固定引用格式：

```markdown
![图 5 车牌识别六阶段流水线](docs/img/pipeline.png)
（draw.io 源文件：docs/img/src/pipeline.drawio）
```

这里是将来导出文件的引用示例；本批不创建 pipeline.png。素材来自仿真数据，不能用于证明正式识别指标；真实 CCPD 到位后，使用相同 viz 流程重新选图、替换 stages/ 素材并同步更新本说明中的来源与尺寸。
