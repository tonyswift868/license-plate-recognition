# 字符模型训练工具

所有命令在仓库根目录执行，使用本工位 `.venv/Scripts/python.exe`。
不下载数据，不使用预训练权重。字体合成数据只用于验证工程流程，
其验证准确率不能代替 CCPD 或真实分割字符上的正式验收。

## 数据

`synth_chars --out algorithm/data/chars_synth --per-class 200 --seed 42`
按 65 类各生成 200 张，默认每类 train 160 / val 40。
`--font` 默认 `C:/Windows/Fonts/simhei.ttf`，`--val-fraction` 默认 0.2。
输出目录必须为空，避免重跑参数改变后残留旧样本；保留已有结果时请另选目录。

结构为 `{train,val}/<类下标>/*.png`，所有图都是已经过共享
`unify_polarity`、`normalize_char` 和合成增广的 32×32 uint8 灰度图。
训练器不再次归一化缓存图片；CNN 训练阶段再在线增广，验证阶段不在线增广。
共享读取器要求 train、val 均包含全部 65 类并检查图片规格。

`chars_index.csv` 为 UTF-8，字段：
`path,split,label,char,plate_color,source,font,seed`。
path 相对数据根目录；label 是冻结字符表下标；source 为 font_synthetic。
训练和验证分别使用独立随机流，但共享字体，因此不是独立实景数据分布。

## CNN

`train_cnn --data algorithm/data/chars_synth --epochs 2 --out algorithm/out/models/char_cnn_synth.pt --version char_cnn_synth`

默认 epochs=15、batch-size=256、lr=0.001、weight-decay=0.0001、seed=42、
num-workers=0（可选 0–2）。结构、Dropout 0.3、AdamW、余弦退火、
label smoothing 0.05 来自开发文档 8.1–8.2。
类别权重为 `N / (65 * class_count)`，每轮全部训练样本遍历一次。
CPU Torch 线程数为 2，OpenCV 为 1；random / NumPy / Torch 均固定种子，
在线增广由 seed、epoch、index 确定，多进程通过 main guard 支持 Windows。

默认日志为模型同名 `.train.csv`（可用 `--log` 指定），字段为
`epoch,loss,val_char_acc,lr`，每轮立即刷新。
最终模型只保存 CPU state_dict；manifest 的 val_char_acc 对应最终模型，
不把优化器或整模型 pickle 写入 CNN 文件。

增广初值：仿射范围遵循 8.2；模糊概率 0.3、σ 0.2–1.2；
噪声概率 0.5、标准差 1–8；对比度 0.85–1.15、亮度偏移 ±15；
半径 1 px 腐蚀/膨胀概率 0.2；2–5 px 遮挡概率 0.2。
概率和未由文档固定的数值是合成冒烟初值，后续须用真实训练/验证集调节。

## 基线

`build_templates --data algorithm/data/chars_synth --out algorithm/out/models/templates_synth.npz --version templates_synth`
按训练类别取 float32 均值图，用 OpenCV `TM_CCOEFF_NORMED` 匹配，
温度 softmax 默认 `--temperature 0.1`（待真实验证集校准）。

`train_svm --data algorithm/data/chars_synth --out algorithm/out/models/svm_synth.joblib --version svm_synth`
使用 `cv2.HOGDescriptor((32,32),(16,16),(8,8),(8,8),9)` 的 324 维特征，
`SVC(kernel="rbf", C=10, probability=True, random_state=42)`，gamma 使用
scikit-learn 默认的 `scale`。默认每类最多 500 张，可用 `--max-per-class`
调低但不可超出 500；抽样由 `--seed` 控制，默认 42。
两种模型都在独立 val 目录计算概率 argmax 准确率并登记 manifest。
运行时每次调用独立创建 HOGDescriptor，模型只读，支持并发推理。

OpenCV 必须提供 `cv2.HOGDescriptor`。工位最初预装的 OpenCV 5.0.0
不提供此 API；代码明确报错，不以其他 HOG 实现替代规定特征。
已按用户明确授权，仅在工位 01 的 `.venv` 安装
`opencv-python==4.14.0.94`，NumPy 保持 2.5.4，全量测试通过。
不修改本分支 requirements；统一依赖约束由架构师维护。
当前 scikit-learn 1.9.1 会提示 `SVC(probability=True)` 将在 1.11 移除；
本批按派工采用此选项，未来升级时可切换文档允许的 `CalibratedClassifierCV`。

## CCPD 字符样本生成

```powershell
.venv\Scripts\python.exe -m lpr_tools.train.build_chars --index algorithm/data/index/ccpd_index.csv --raw algorithm/data/raw --out algorithm/data/chars --exclude-manifest algorithm/testset/manifest.csv
```

输入采用 `lpr_tools.data.ccpd.read_index` 的 CSV 契约。仅使用
`has_plate=1`、`skip_reason` 为空且 split 为 train/val 的记录，
原图 split 保持不变；GT quad 已为 LT、RT、RB、LB，不再交换点序。
按配置透视到 252×80 或 276×80，再按对应版式中心和槽宽切位，
共享 `unify_polarity` 与 `normalize_char`，输出灰度 32×32 PNG。
所有图片读写经 `lpr.io_utils`，支持中文路径。

`--max-per-class` 默认 8000，限制每类 train+val 合计数量；
按两个 split 的可用数量分配额度，预算至少 2 且两边均有样本时各保留至少 1。
cap=1 优先 train。此规则避免索引中 train 在前时耗尽 val 额度。
测试清单 path 支持绝对路径及清单相对路径，也兼容 raw 相对路径。
跨 split 的重复路径或相同文件内容会在写出字符前报错；不自动重分 split。
输出目录须为空，避免旧样本混入新索引；文件名包含原图名、路径短哈希和字符位置，
防止不同目录下同名图片覆盖。

输出 `chars_index.csv` 字段为
`path,source_path,split,label,char,plate_no,position`；字符 path 相对输出目录，
source_path 为原图绝对路径，position 从 0 开始。
`chars_stats.json` 保存每类数量、split 总数、排除图数和每类截断数，
同时向标准输出打印 65 类数量表。
